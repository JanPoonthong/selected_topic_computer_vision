import os
import sys
import logging
import warnings
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
import torchvision.transforms as T
from PIL import Image
import trimesh
from scipy.spatial import cKDTree

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Suppress specific warnings
warnings.filterwarnings(
    "ignore", category=UserWarning, module="timm.models.vision_transformer"
)

# Set environment variables
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"


def load_midas_model():
    logging.info("Loading MiDaS model...")
    model_type = "DPT_Large"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform

    return midas, transform


def estimate_depth(image_path, midas, transform):
    logging.info(f"Estimating depth for image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to load image from {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    midas = midas.to(device)

    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    logging.info(
        f"Depth map shape: {depth_map.shape}, min: {depth_map.min()}, max: {depth_map.max()}"
    )
    return depth_map, img


def create_point_cloud(depth_map, image):
    h, w = depth_map.shape
    y, x = np.mgrid[0:h, 0:w]
    points = np.column_stack([x.ravel(), y.ravel(), depth_map.ravel()])
    colors = image.reshape(-1, 3)
    return points, colors


def load_3d_object(object_file_path):
    logging.info(f"Loading 3D object from {object_file_path}")
    mesh = trimesh.load(object_file_path)
    logging.info(
        f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces"
    )
    return mesh


def icp(A, B, max_iterations=50, tolerance=1e-5):
    logging.info("Starting ICP alignment...")

    def best_fit_transform(A, B):
        assert A.shape == B.shape

        m = A.shape[1]
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B

        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        if np.linalg.det(R) < 0:
            Vt[m - 1, :] *= -1
            R = np.dot(Vt.T, U.T)

        t = centroid_B.T - np.dot(R, centroid_A.T)

        T = np.identity(m + 1)
        T[:m, :m] = R
        T[:m, m] = t

        return T, R, t

    T = np.identity(4)

    src = np.ones((4, A.shape[0]))
    dst = np.ones((4, B.shape[0]))
    src[:3, :] = np.copy(A.T)
    dst[:3, :] = np.copy(B.T)

    prev_error = 0

    for i in range(max_iterations):
        tree = cKDTree(dst[:3, :].T)
        distances, indices = tree.query(src[:3, :].T, k=1)

        T_step, _, _ = best_fit_transform(src[:3, :].T, dst[:3, indices].T)
        src = np.dot(T_step, src)

        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    T, _, _ = best_fit_transform(A, src[:3, :].T)

    logging.info(
        f"ICP completed after {i+1} iterations. Final mean error: {mean_error}"
    )
    return T, distances


def align_3d_object(mesh, points):
    logging.info("Aligning 3D object with point cloud...")

    # Sample points from the mesh
    mesh_points = mesh.sample(points.shape[0])

    # Perform ICP
    T, _ = icp(mesh_points, points)

    # Apply transformation to mesh
    mesh.apply_transform(T)

    return mesh


def visualize_results(image, depth_map, mesh, points, colors):
    logging.info("Visualizing results...")
    fig = plt.figure(figsize=(20, 10))

    # Original Image
    ax1 = fig.add_subplot(231)
    ax1.imshow(image)
    ax1.set_title("Original Image")

    # Depth Map
    ax2 = fig.add_subplot(232)
    depth_map_vis = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    ax2.imshow(depth_map_vis, cmap="viridis")
    ax2.set_title("Depth Map")

    # 3D Point Cloud
    ax3 = fig.add_subplot(233, projection="3d")
    ax3.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors / 255, s=1)
    ax3.set_title("3D Point Cloud")

    # 3D Visualization with Mesh
    ax4 = fig.add_subplot(212, projection="3d")
    mesh_plot = ax4.add_collection3d(
        Poly3DCollection(mesh.triangles, alpha=0.3, edgecolor="r")
    )
    ax4.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors / 255, s=1)
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    ax4.set_zlabel("Z")
    ax4.set_title("3D Visualization with Aligned Mesh")

    # Set consistent limits for 3D plots
    max_range = (
        np.array(
            [
                points[:, 0].max() - points[:, 0].min(),
                points[:, 1].max() - points[:, 1].min(),
                points[:, 2].max() - points[:, 2].min(),
            ]
        ).max()
        / 2.0
    )
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    ax3.set_xlim(mid_x - max_range, mid_x + max_range)
    ax3.set_ylim(mid_y - max_range, mid_y + max_range)
    ax3.set_zlim(mid_z - max_range, mid_z + max_range)
    ax4.set_xlim(mid_x - max_range, mid_x + max_range)
    ax4.set_ylim(mid_y - max_range, mid_y + max_range)
    ax4.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show()


def main(image_path, object_file_path):
    try:
        midas, transform = load_midas_model()
        depth_map, image = estimate_depth(image_path, midas, transform)

        # Create point cloud from depth map and image
        points, colors = create_point_cloud(depth_map, image)

        mesh = load_3d_object(object_file_path)
        aligned_mesh = align_3d_object(mesh, points)

        visualize_results(image, depth_map, aligned_mesh, points, colors)

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.error("Please check your environment setup and input files.")
        raise


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <image_path> <object_file_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    object_file_path = sys.argv[2]
    main(image_path, object_file_path)