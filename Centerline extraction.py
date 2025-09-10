import nibabel as nib
import numpy as np
from skimage import morphology, measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree

def extract_centerline(nii_file):
 
    img = nib.load(nii_file)
    data = img.get_fdata()


    smoothed = morphology.binary_closing(data, morphology.ball(3))


    skeleton = morphology.skeletonize_3d(smoothed)

 
    labeled_skeleton = measure.label(skeleton)
    props = measure.regionprops(labeled_skeleton)
    large_components = [prop.label for prop in props if prop.area > 10]


    centerline = np.isin(labeled_skeleton, large_components)

    return centerline

def find_nearest_neighbors(coords):

    kdtree = cKDTree(coords)
    indices = kdtree.query(coords, k=2)[1][:, 1]
    return indices

def plot_3d_centerline(centerline):

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    coords = np.array(np.where(centerline)).T
    nearest_neighbors = find_nearest_neighbors(coords)
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c='r', marker='o', s=0.1)
    for i in range(len(coords)):
        ax.plot([coords[i, 0], coords[nearest_neighbors[i], 0]],
                [coords[i, 1], coords[nearest_neighbors[i], 1]],
                [coords[i, 2], coords[nearest_neighbors[i], 2]], c='r', linewidth=1)


    ax.axis('off')


    plt.show()

if __name__ == "__main__":
    nii_file_path = '/Volumes/T7 Shield/wangyuan/06_dataset/TrainBatch1/labelsTr/ATM_011_0000.nii.gz'
    

    centerline = extract_centerline(nii_file_path)


    plot_3d_centerline(centerline)


    output_nii_path = '/Users/wangyuan/Desktop/centerline.nii'  # 替换为你想要保存的文件路径和名称
    centerline = centerline.astype(np.uint8)  # 将数据类型转换为uint8
    img = nib.load(nii_file_path)  # 加载原始图像
    centerline_nii = nib.Nifti1Image(centerline, img.affine)  # 使用原始图像的仿射变换矩阵


    nib.save(centerline_nii, output_nii_path)

    print(f"Centerline saved as {output_nii_path}")
