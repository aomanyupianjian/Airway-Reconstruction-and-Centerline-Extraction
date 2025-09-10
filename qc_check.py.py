import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Optional

try:
    import numpy as np
except Exception as e:
    raise RuntimeError("NumPy is required. Install with `pip install numpy`.") from e

_nib_ok = True
try:
    import nibabel as nib
except Exception:
    _nib_ok = False
    try:
        import SimpleITK as sitk
    except Exception as e:
        raise RuntimeError("Install `nibabel` or `SimpleITK`.") from e

try:
    from skimage.morphology import skeletonize_3d
except Exception as e:
    raise RuntimeError("scikit-image is required. Install with `pip install scikit-image`.") from e

try:
    import networkx as nx
except Exception as e:
    raise RuntimeError("networkx is required. Install with `pip install networkx`.") from e

try:
    import yaml
except Exception as e:
    raise RuntimeError("PyYAML is required. Install with `pip install pyyaml`.") from e


def _read_mask(path: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    if _nib_ok and path.lower().endswith(('.nii', '.nii.gz')):
        img = nib.load(path)
        data = img.get_fdata()
        aff = img.affine
        sx = float(np.linalg.norm(aff[:3, 0]))
        sy = float(np.linalg.norm(aff[:3, 1]))
        sz = float(np.linalg.norm(aff[:3, 2]))
        spacing = (sx, sy, sz)
        arr = np.asarray(data, dtype=np.float32)
    else:
        itk_img = sitk.ReadImage(path)
        spacing_sitk = tuple(map(float, itk_img.GetSpacing()))
        data = sitk.GetArrayFromImage(itk_img)
        arr = np.transpose(np.asarray(data, dtype=np.float32), (2, 1, 0))
        spacing = spacing_sitk
    mask = (arr > 0.5)
    return mask.astype(bool), spacing


def _bbox(mask: np.ndarray):
    coords = np.where(mask)
    if len(coords[0]) == 0:
        return slice(0, 1), slice(0, 1), slice(0, 1)
    x0, x1 = coords[0].min(), coords[0].max() + 1
    y0, y1 = coords[1].min(), coords[1].max() + 1
    z0, z1 = coords[2].min(), coords[2].max() + 1
    return slice(x0, x1), slice(y0, y1), slice(z0, z1)


def _neighbors26(x, y, z):
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                yield (dx, dy, dz)


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    return skeletonize_3d(mask.astype(np.uint8)) > 0


def _build_graph_from_skeleton(skel: np.ndarray, spacing: Tuple[float, float, float]) -> nx.Graph:
    G = nx.Graph()
    xs, ys, zs = np.where(skel)
    for x, y, z in zip(xs, ys, zs):
        G.add_node((int(x), int(y), int(z)))
    sx, sy, sz = spacing
    for x, y, z in zip(xs, ys, zs):
        for dx, dy, dz in _neighbors26(x, y, z):
            nx_, ny_, nz_ = x + dx, y + dy, z + dz
            if G.has_node((nx_, ny_, nz_)):
                w = math.sqrt((dx * sx) ** 2 + (dy * sy) ** 2 + (dz * sz) ** 2)
                G.add_edge((x, y, z), (nx_, ny_, nz_), weight=w)
    return G


def _prune_short_spurs(G: nx.Graph, tau_spur_mm: float) -> Tuple[nx.Graph, float]:
    G = G.copy()
    total_edges_before = G.number_of_edges()
    removed_edges = 0
    changed = True
    while changed:
        changed = False
        leaves = [n for n in list(G.nodes) if G.degree(n) == 1]
        for leaf in leaves:
            path = [leaf]
            cur = leaf
            prev = None
            path_len = 0.0
            while True:
                nbrs = [n for n in G.neighbors(cur) if n != prev] if prev is not None else list(G.neighbors(cur))
                if len(nbrs) != 1:
                    break
                nxt = nbrs[0]
                w = G[cur][nxt]['weight']
                path_len += w
                path.append(nxt)
                prev, cur = cur, nxt
                if G.degree(cur) != 2:
                    break
            if path_len < tau_spur_mm:
                for i in range(len(path) - 1):
                    if G.has_edge(path[i], path[i + 1]):
                        G.remove_edge(path[i], path[i + 1])
                        removed_edges += 1
                for n in path:
                    if G.degree(n) == 0:
                        G.remove_node(n)
                changed = True
    ratio = (removed_edges / total_edges_before) if total_edges_before > 0 else 0.0
    return G, ratio


def _components_by_length(G: nx.Graph) -> List[Tuple[float, List[Tuple[int, int, int]]]]:
    comps = []
    for nodes in nx.connected_components(G):
        sub = G.subgraph(nodes)
        total_len = sum(d.get('weight', 1.0) for _, _, d in sub.edges(data=True))
        comps.append((total_len, list(nodes)))
    comps.sort(key=lambda x: x[0], reverse=True)
    return comps


def _lung_height_mm(mask: np.ndarray, spacing: Tuple[float, float, float]) -> float:
    s_x, s_y, s_z = _bbox(mask)
    return (s_z.stop - s_z.start) * spacing[2]


def _approx_longest_shortest_path_mm(G: nx.Graph) -> float:
    if G.number_of_nodes() == 0:
        return 0.0
    nodes = list(G.nodes())
    start = nodes[0]
    dist1 = nx.single_source_dijkstra_path_length(G, start, weight='weight')
    a = max(dist1.items(), key=lambda x: x[1])[0]
    dist2 = nx.single_source_dijkstra_path_length(G, a, weight='weight')
    return max(dist2.values()) if dist2 else 0.0


def _estimate_generation_depth(G: nx.Graph) -> Tuple[int, int]:
    if G.number_of_nodes() == 0:
        return 0, 0
    endpoints = [n for n in G.nodes if G.degree(n) == 1]
    if not endpoints:
        return 0, 0
    top = min(endpoints, key=lambda p: p[2])
    tree = nx.bfs_tree(G, source=top)
    max_depth = 0
    xs = [n[0] for n in G.nodes]
    x_mid = float(np.median(xs))
    per_lung_leaves = [0, 0]
    for n in tree.nodes():
        path = nx.shortest_path(G, top, n)
        depth = sum(1 for v in path if G.degree(v) >= 3)
        if depth > max_depth:
            max_depth = depth
    for e in endpoints:
        side = 0 if e[0] < x_mid else 1
        per_lung_leaves[side] += 1
    per_lung_leaf_min = int(min(per_lung_leaves))
    return int(max_depth), per_lung_leaf_min


def _pick_carina_candidate(G: nx.Graph) -> Optional[Tuple[int, int, int]]:
    branch_nodes = [n for n in G.nodes if G.degree(n) >= 3]
    if not branch_nodes:
        return None
    zs = np.array([n[2] for n in G.nodes], dtype=float)
    z_half = np.median(zs)
    pool = [n for n in branch_nodes if n[2] <= z_half] or branch_nodes
    pool.sort(key=lambda n: (G.degree(n), sum(G.degree(nb) for nb in G.neighbors(n))), reverse=True)
    return pool[0] if pool else None


@dataclass
class Metrics:
    case_id: str
    decision: str
    n_components: int
    second_comp_share: float
    carina_exists: int
    carina_to_zborder_mm: float
    max_generation_depth: int
    per_lung_leaf_min: int
    Lmax_over_H: float
    L_skeleton_mm: float
    short_spur_ratio: float


def _decide(m: Metrics, th: Dict) -> str:
    fail = False
    borderline = False
    if m.n_components > 1 and m.second_comp_share >= th['second_comp_share_fail']:
        fail = True
    elif m.n_components > 1 and m.second_comp_share >= th['second_comp_share_border']:
        borderline = True
    if m.carina_exists == 0:
        fail = True
    elif m.carina_to_zborder_mm < th['carina_to_zborder_border_mm']:
        borderline = True
    if m.max_generation_depth < th['min_generation_depth_fail']:
        fail = True
    elif (m.max_generation_depth == th['min_generation_depth_fail'] and m.per_lung_leaf_min < th['per_lung_leaf_min_border']):
        borderline = True
    if m.Lmax_over_H < th['Lmax_over_H_fail']:
        fail = True
    elif m.Lmax_over_H < th['Lmax_over_H_border']:
        borderline = True
    if m.L_skeleton_mm < th['L_skeleton_mm_fail']:
        fail = True
    elif m.L_skeleton_mm < th['L_skeleton_mm_border']:
        borderline = True
    if m.short_spur_ratio > th['short_spur_ratio_fail']:
        fail = True
    elif m.short_spur_ratio > th['short_spur_ratio_border']:
        borderline = True
    if fail:
        return "FAIL"
    if borderline:
        return "BORDERLINE"
    return "PASS"


def run_qc(input_dir: str, pattern: str, out_dir: str, cfg_path: str, save_graphs: bool = False, log_each: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    graphs_dir = os.path.join(out_dir, 'graphs')
    logs_dir = os.path.join(out_dir, 'logs')
    if save_graphs:
        os.makedirs(graphs_dir, exist_ok=True)
    if log_each:
        os.makedirs(logs_dir, exist_ok=True)
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    tau_spur_mm = float(cfg['preprocess']['tau_spur_mm'])
    th = cfg['thresholds']
    import glob
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        raise RuntimeError(f"No files matched: {os.path.join(input_dir, pattern)}")
    report_csv = os.path.join(out_dir, 'qc_report.csv')
    with open(report_csv, 'w', newline='') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(['case_id','decision','n_components','second_comp_share','carina_exists','carina_to_zborder_mm','max_generation_depth','per_lung_leaf_min','Lmax_over_H','L_skeleton_mm','short_spur_ratio'])
        fail_list, borderline_list = [], []
        for fp in files:
            case_id = os.path.splitext(os.path.basename(fp).replace('.nii', '').replace('.gz',''))[0]
            try:
                mask, spacing = _read_mask(fp)
                if mask.sum() == 0:
                    raise ValueError("Empty mask.")
                s_x, s_y, s_z = _bbox(mask)
                mask_c = mask[s_x, s_y, s_z]
                skel = _skeletonize(mask_c)
                G = _build_graph_from_skeleton(skel, spacing)
                Gp, spur_ratio = _prune_short_spurs(G, tau_spur_mm=tau_spur_mm)
                comps = _components_by_length(Gp)
                n_comp = len(comps)
                total_len = sum(c[0] for c in comps) if comps else 0.0
                second_share = (comps[1][0] / total_len) if len(comps) > 1 and total_len > 0 else 0.0
                if comps:
                    nodes_main = set(comps[0][1])
                    Gmain = Gp.subgraph(nodes_main).copy()
                else:
                    Gmain = Gp
                H_lung = _lung_height_mm(mask, spacing)
                carina = _pick_carina_candidate(Gmain)
                carina_exists = 1 if carina is not None else 0
                carina_to_border_mm = 0.0
                if carina is not None:
                    cz_local = carina[2]
                    cz_full = cz_local + s_z.start
                    dz_vox = min(cz_full, (mask.shape[2] - 1) - cz_full)
                    carina_to_border_mm = dz_vox * spacing[2]
                Lmax = _approx_longest_shortest_path_mm(Gmain)
                L_skel = sum(d.get('weight', 1.0) for _, _, d in Gmain.edges(data=True))
                Lmax_over_H = (Lmax / H_lung) if H_lung > 0 else 0.0
                max_depth, leaf_min = _estimate_generation_depth(Gmain)
                m = Metrics(case_id=case_id, decision="UNKNOWN", n_components=n_comp, second_comp_share=second_share, carina_exists=carina_exists, carina_to_zborder_mm=carina_to_border_mm, max_generation_depth=max_depth, per_lung_leaf_min=leaf_min, Lmax_over_H=Lmax_over_H, L_skeleton_mm=L_skel, short_spur_ratio=spur_ratio)
                m.decision = _decide(m, th)
                writer.writerow([m.case_id, m.decision, m.n_components, f"{m.second_comp_share:.4f}", m.carina_exists, f"{m.carina_to_zborder_mm:.1f}", m.max_generation_depth, m.per_lung_leaf_min, f"{m.Lmax_over_H:.3f}", f"{m.L_skeleton_mm:.1f}", f"{m.short_spur_ratio:.3f}"])
                if m.decision.startswith("FAIL"):
                    fail_list.append(case_id)
                elif m.decision == "BORDERLINE":
                    borderline_list.append(case_id)
                if save_graphs and Gmain.number_of_nodes() > 0:
                    nx.write_gpickle(Gmain, os.path.join(graphs_dir, f"{case_id}_graph.gpickle"))
                if log_each:
                    with open(os.path.join(logs_dir, f"{case_id}.txt"), 'w') as flog:
                        flog.write(json.dumps(asdict(m), indent=2))
            except Exception as e:
                if log_each:
                    with open(os.path.join(logs_dir, f"{case_id}_ERROR.txt"), 'w') as flog:
                        flog.write(str(e))
                writer.writerow([case_id, "FAIL(ERROR)", "", "", "", "", "", "", "", "", ""])
                fail_list.append(case_id)
    with open(os.path.join(out_dir, 'exclusion_list.txt'), 'w') as fex:
        fex.write("\n".join(fail_list))
    with open(os.path.join(out_dir, 'borderline_list.txt'), 'w') as fbd:
        fbd.write("\n".join(borderline_list))
    print(f"[QC] Done: {report_csv}")


def main():
    parser = argparse.ArgumentParser(description="ATM22 Airway QC (algorithmic)")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with airway masks')
    parser.add_argument('--pattern', type=str, default='*.nii.gz', help='Glob pattern, e.g., *.nii.gz')
    parser.add_argument('--config', type=str, required=True, help='YAML config file')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--save_graphs', action='store_true', help='Save skeleton graphs')
    parser.add_argument('--log_each', action='store_true', help='Write per-case logs')
    args = parser.parse_args()
    run_qc(input_dir=args.input_dir, pattern=args.pattern, out_dir=args.out_dir, cfg_path=args.config, save_graphs=args.save_graphs, log_each=args.log_each)


if __name__ == '__main__':
    main()
