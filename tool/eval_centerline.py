#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centerline evaluation: length-weighted, tolerance-based matching.
- Resample predicted/GT centerlines to fixed step (e.g., 1 mm)
- Evaluable region: lung_mask AND dilated GT airway mask (1 voxel)
- Matching: nearest-neighbor distance threshold tau (mm)
- Radius-aware tube: accept if distance <= (local airway radius + delta)
- Metrics: OR, length-precision/recall, FP-length ratio, AD, MD
- Fragment suppression: drop predicted polylines < frag_mm
- In-mask containment: points outside evaluable region -> FP

Inputs (centerlines):
  Recommended CSV with columns: x,y,z,path_id[,order]
  - Units: millimeters in the same world space as the masks.
  - Points should be ordered along each polyline (if 'order' absent, current row order is used).
  - Use one row per vertex. Multiple polylines distinguished by path_id.

Masks:
  - NIfTI (.nii/.nii.gz) for lung_mask and GT airway_mask.
  - Must be aligned (same shape + affine). World coordinates must match the CSV points.

Usage:
  python eval_centerline.py \
    --pred_csv pred_centerline.csv \
    --gt_csv gt_centerline.csv \
    --lung_mask lung_mask.nii.gz \
    --gt_mask gt_airway_mask.nii.gz \
    --out_dir ./eval_out \
    --tau 2.0 --delta 1.5 --resample_mm 1.0 --frag_mm 3.0 --dilate_vox 1
"""

import os
import json
import math
import argparse
from typing import Dict, Tuple, List

import numpy as np
try:
    import pandas as pd
except Exception as e:
    raise RuntimeError("pandas is required: pip install pandas") from e

try:
    import nibabel as nib
except Exception as e:
    raise RuntimeError("nibabel is required: pip install nibabel") from e

try:
    from scipy.ndimage import distance_transform_edt, binary_dilation, map_coordinates
    from scipy.spatial import cKDTree
except Exception as e:
    raise RuntimeError("scipy is required: pip install scipy") from e


# ---------------------------
# I/O helpers
# ---------------------------
def load_mask(path: str) -> Tuple[np.ndarray, np.ndarray]:
    img = nib.load(path)
    data = img.get_fdata().astype(np.uint8)
    aff = img.affine
    return data, aff


def check_alignment(mask_a: np.ndarray, aff_a: np.ndarray,
                    mask_b: np.ndarray, aff_b: np.ndarray):
    if mask_a.shape != mask_b.shape:
        raise ValueError(f"Mask shape mismatch: {mask_a.shape} vs {mask_b.shape}")
    if not np.allclose(aff_a, aff_b, atol=1e-5):
        raise ValueError("Mask affines differ; please align masks beforehand.")


def world_to_vox(aff: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Map Nx3 world (mm) -> voxel ijk (float)."""
    if xyz.ndim == 1:
        xyz = xyz[None, :]
    inv = np.linalg.inv(aff)
    homo = np.c_[xyz, np.ones((xyz.shape[0], 1))]
    ijk = homo @ inv.T
    return ijk[:, :3]


def vox_to_world(aff: np.ndarray, ijk: np.ndarray) -> np.ndarray:
    if ijk.ndim == 1:
        ijk = ijk[None, :]
    homo = np.c_[ijk, np.ones((ijk.shape[0], 1))]
    xyz = homo @ aff.T
    return xyz[:, :3]


def load_centerline_csv(path: str) -> Dict[int, np.ndarray]:
    """
    Load CSV with columns: x,y,z,path_id[,order]
    Returns dict: path_id -> (N_i,3) float32, in given row order or sorted by 'order' if present.
    """
    df = pd.read_csv(path)
    if not {"x", "y", "z", "path_id"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: x,y,z,path_id[,order]")
    pid_groups = {}
    if "order" in df.columns:
        df = df.sort_values(by=["path_id", "order"])
    for pid, g in df.groupby("path_id"):
        pts = g[["x", "y", "z"]].to_numpy(dtype=np.float32)
        pid_groups[int(pid)] = pts
    return pid_groups


# ---------------------------
# Geometry helpers
# ---------------------------
def polyline_length(pts: np.ndarray) -> float:
    if pts.shape[0] < 2:
        return 0.0
    dif = np.diff(pts, axis=0)
    seg = np.linalg.norm(dif, axis=1)
    return float(seg.sum())


def resample_polyline(pts: np.ndarray, step: float) -> np.ndarray:
    """
    Uniformly resample a 3D polyline to a given step length (mm).
    Returns new (M,3) array. Keeps first and last points (if length > 0).
    """
    if pts.shape[0] == 0:
        return pts
    if pts.shape[0] == 1:
        return pts.copy()

    # cumulative arc length
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    L = float(seg.sum())
    if L == 0:
        return pts[[0], :].copy()

    n_steps = max(1, int(round(L / step)))
    target_s = np.linspace(0.0, L, n_steps + 1)

    # interpolate
    new_pts = [pts[0]]
    acc = 0.0
    i = 0
    for t in target_s[1:]:
        # advance along original segments until reaching arc length t
        while i < len(seg) and acc + seg[i] < t:
            acc += seg[i]
            i += 1
        if i >= len(seg):
            new_pts.append(pts[-1])
            continue
        r = (t - acc) / max(seg[i], 1e-8)
        p = (1 - r) * pts[i] + r * pts[i + 1]
        new_pts.append(p)
    return np.vstack(new_pts)


def concat_resampled(centerlines: Dict[int, np.ndarray], step: float,
                     frag_mm: float) -> np.ndarray:
    """
    Resample each polyline to step; drop polylines with length < frag_mm.
    Concatenate into a single (N,3) array of sample points (ordered within each polyline).
    """
    out = []
    for pid, pts in centerlines.items():
        L = polyline_length(pts)
        if L < frag_mm:
            continue
        rs = resample_polyline(pts, step)
        out.append(rs)
    if not out:
        return np.zeros((0, 3), dtype=np.float32)
    return np.vstack(out).astype(np.float32)


# ---------------------------
# Evaluation core
# ---------------------------
def build_evaluable_region(lung: np.ndarray, gt_air: np.ndarray, dilate_vox: int) -> np.ndarray:
    if dilate_vox > 0:
        from scipy.ndimage import generate_binary_structure
        st = generate_binary_structure(3, 1)  # 6/18/26? 1->6-connectivity, adequate for a 1-voxel pad
        gt_dil = binary_dilation(gt_air.astype(bool), structure=st, iterations=int(dilate_vox))
    else:
        gt_dil = gt_air.astype(bool)
    region = (lung.astype(bool) & gt_dil)
    return region.astype(bool)


def sample_volume_at_world(volume: np.ndarray, aff: np.ndarray, xyz: np.ndarray, order: int = 1) -> np.ndarray:
    """
    Trilinear sample 'volume' at world coords xyz (N,3) -> values (N,).
    map_coordinates expects index order (z,y,x), so we must pass coords accordingly.
    """
    ijk = world_to_vox(aff, xyz)  # (i,j,k)
    coords = np.vstack([ijk[:, 2], ijk[:, 1], ijk[:, 0]])  # (z,y,x)
    vals = map_coordinates(volume, coords, order=order, mode='nearest')
    return vals


def in_region_mask(region: np.ndarray, aff: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    vals = sample_volume_at_world(region.astype(np.uint8), aff, xyz, order=0)
    return vals > 0.5


def nearest_distances(src_xyz: np.ndarray, ref_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each src point, compute (distance_mm, index_in_ref) to nearest ref point.
    """
    if ref_xyz.shape[0] == 0 or src_xyz.shape[0] == 0:
        return np.array([]), np.array([], dtype=int)
    tree = cKDTree(ref_xyz)
    d, idx = tree.query(src_xyz, k=1, workers=-1)
    return d.astype(np.float32), idx.astype(np.int64)


def radius_edt_from_mask(gt_air: np.ndarray, spacing_xyz: Tuple[float, float, float]) -> np.ndarray:
    """
    Distance transform INSIDE the GT airway mask, in millimeters.
    For a voxel inside the airway, value approximates local radius to airway wall.
    """
    # SciPy EDT returns distance to the nearest zero; pass mask==1 to measure distance to background (wall)
    edt = distance_transform_edt(gt_air.astype(bool), sampling=spacing_xyz)
    return edt.astype(np.float32)


def spacing_from_affine(aff: np.ndarray) -> Tuple[float, float, float]:
    sx = float(np.linalg.norm(aff[:3, 0]))
    sy = float(np.linalg.norm(aff[:3, 1]))
    sz = float(np.linalg.norm(aff[:3, 2]))
    return (sx, sy, sz)


def evaluate_centerlines(pred_xyz: np.ndarray,
                         gt_xyz: np.ndarray,
                         region: np.ndarray,
                         gt_air: np.ndarray,
                         aff: np.ndarray,
                         tau_mm: float,
                         delta_mm: float,
                         step_mm: float) -> Dict[str, float]:
    """
    pred_xyz, gt_xyz: (N,3) world-mm points (resampled)
    region: evaluable region (bool) in voxel space (same affine as gt_air)
    gt_air: GT airway mask (bool) in voxel space
    aff: affine of masks
    Returns dict of metrics and raw lengths.
    """
    # Limit predicted to region membership (for counting total predicted length in-region)
    pred_in_region = in_region_mask(region, aff, pred_xyz)
    pred_xyz_in = pred_xyz[pred_in_region]
    pred_xyz_out = pred_xyz[~pred_in_region]  # counted as FP by definition

    # Nearest distances from pred->gt
    d_pred2gt, idx_pred2gt = nearest_distances(pred_xyz_in, gt_xyz)

    # Local radius at matched GT points via EDT sampled at those GT points
    spacing = spacing_from_affine(aff)
    edt_mm = radius_edt_from_mask(gt_air, spacing)
    # sample EDT at world coords of the matched GT points
    gt_matched_xyz = gt_xyz[idx_pred2gt] if idx_pred2gt.size > 0 else np.zeros((0, 3), dtype=np.float32)
    r_local = sample_volume_at_world(edt_mm, aff, gt_matched_xyz, order=1) if gt_matched_xyz.shape[0] > 0 else np.array([])

    # Acceptance threshold: max(tau, r_local + delta)
    thr = np.maximum(tau_mm, r_local + delta_mm) if r_local.size > 0 else np.array([])

    # Matched predicted samples = inside region AND distance <= threshold
    matched_pred_mask = np.zeros(pred_xyz.shape[0], dtype=bool)
    # mark in-region matched
    if pred_xyz_in.shape[0] > 0:
        matched_in = d_pred2gt <= thr
        # write back to original indexing
        matched_pred_mask[np.where(pred_in_region)[0][matched_in]] = True

    # Length bookkeeping (each sample counts as step_mm)
    # L_pred includes only in-region points (consistent with FP-length ratio denominator choice)
    L_pred = pred_xyz_in.shape[0] * step_mm
    L_fp_outside = pred_xyz_out.shape[0] * step_mm  # these are FP by definition
    L_tp = matched_pred_mask.sum() * step_mm
    L_fp_in = (pred_xyz_in.shape[0] - (matched_pred_mask[pred_in_region].sum() if pred_in_region.any() else 0)) * step_mm
    L_fp = L_fp_in + L_fp_outside

    # False negatives: GT samples not matched by any pred within threshold (symmetric check)
    d_gt2pred, idx_gt2pred = nearest_distances(gt_xyz, pred_xyz)
    # For each GT sample, build local threshold max(tau, r_local + delta) using same EDT
    r_gt_local = sample_volume_at_world(edt_mm, aff, gt_xyz, order=1) if gt_xyz.shape[0] > 0 else np.array([])
    thr_gt = np.maximum(tau_mm, r_gt_local + delta_mm) if r_gt_local.size > 0 else np.array([])
    matched_gt_mask = (d_gt2pred <= thr_gt) if d_gt2pred.size > 0 else np.array([], dtype=bool)
    L_fn = (gt_xyz.shape[0] - matched_gt_mask.sum()) * step_mm

    # Distances for AD/MD: only on matched predicted samples
    d_matched = d_pred2gt[matched_pred_mask[pred_in_region]] if pred_xyz_in.shape[0] > 0 else np.array([])
    AD = float(np.mean(d_matched)) if d_matched.size > 0 else float("nan")
    MD = float(np.max(d_matched)) if d_matched.size > 0 else float("nan")

    # OR, precision_l, recall_l, FP-length ratio
    OR = float(L_tp / (L_tp + L_fp + L_fn)) if (L_tp + L_fp + L_fn) > 0 else 0.0
    precision_l = float(L_tp / (L_tp + L_fp)) if (L_tp + L_fp) > 0 else 0.0
    recall_l = float(L_tp / (L_tp + L_fn)) if (L_tp + L_fn) > 0 else 0.0
    fp_len_ratio = float(L_fp / L_pred) if L_pred > 0 else float("nan")

    return {
        "L_TP": float(L_tp),
        "L_FP": float(L_fp),
        "L_FN": float(L_fn),
        "L_pred_inregion": float(L_pred),
        "OR": OR,
        "precision_l": precision_l,
        "recall_l": recall_l,
        "FP_length_ratio": fp_len_ratio,
        "AD": AD,
        "MD": MD,
        "tau_mm": float(tau_mm),
        "delta_mm": float(delta_mm),
        "step_mm": float(step_mm),
    }


# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Length-weighted tolerance-based centerline evaluation")
    ap.add_argument("--pred_csv", required=True, type=str, help="Predicted centerline CSV (x,y,z,path_id[,order])")
    ap.add_argument("--gt_csv", required=True, type=str, help="GT centerline CSV (x,y,z,path_id[,order])")
    ap.add_argument("--lung_mask", required=True, type=str, help="Lung mask NIfTI")
    ap.add_argument("--gt_mask", required=True, type=str, help="GT airway mask NIfTI")
    ap.add_argument("--out_dir", required=True, type=str, help="Output directory")
    ap.add_argument("--tau", type=float, default=2.0, help="Distance tolerance tau (mm)")
    ap.add_argument("--delta", type=float, default=1.5, help="Radius-aware margin delta (mm)")
    ap.add_argument("--resample_mm", type=float, default=1.0, help="Resampling step (mm)")
    ap.add_argument("--frag_mm", type=float, default=3.0, help="Drop predicted polylines shorter than this (mm)")
    ap.add_argument("--dilate_vox", type=int, default=1, help="GT airway mask dilation (voxels) for evaluable region")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load masks
    lung, aff_lung = load_mask(args.lung_mask)
    gt_air, aff_air = load_mask(args.gt_mask)
    check_alignment(lung, aff_lung, gt_air, aff_air)
    aff = aff_air  # common affine

    # Evaluable region
    region = build_evaluable_region(lung, gt_air, args.dilate_vox)

    # Load centerlines and resample
    pred_cls = load_centerline_csv(args.pred_csv)
    gt_cls = load_centerline_csv(args.gt_csv)

    pred_xyz = concat_resampled(pred_cls, step=args.resample_mm, frag_mm=args.frag_mm)
    gt_xyz = concat_resampled(gt_cls, step=args.resample_mm, frag_mm=0.0)  # do not drop GT

    # Evaluate
    metrics = evaluate_centerlines(
        pred_xyz=pred_xyz,
        gt_xyz=gt_xyz,
        region=region,
        gt_air=gt_air.astype(bool),
        aff=aff,
        tau_mm=args.tau,
        delta_mm=args.delta,
        step_mm=args.resample_mm,
    )

    # Save JSON + CSV
    out_json = os.path.join(args.out_dir, "metrics.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    out_csv = os.path.join(args.out_dir, "metrics.csv")
    with open(out_csv, "w") as f:
        keys = list(metrics.keys())
        f.write(",".join(keys) + "\n")
        f.write(",".join(str(metrics[k]) for k in keys) + "\n")

    print(f"[OK] Saved metrics to:\n  {out_json}\n  {out_csv}")


if __name__ == "__main__":
    main()
