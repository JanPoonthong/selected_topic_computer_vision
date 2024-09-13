import cv2
import numpy as np
import open3d as o3d

# Step 1: Read the Depth Map
def read_depth_map(file_path):
    # Load the depth map image as grayscale
    depth_map = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if depth_map is None:
        raise FileNotFoundError(f"Unable to load image at {file_path}")
        
    depth_map = cv2.normalize(depth_map.astype(np.float32), None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX)  # Normalize to range 0-1
    return depth_map

# Step 2: Convert Depth Map to Point Cloud
def depth_map_to_point_cloud(depth_map, scale=1.0):
    h, w = depth_map.shape  # Ensure depth_map has only two dimensions (height and width)
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_map * scale  # Scale depth values

    # Create a 3D point cloud from depth values
    points_3d = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    return points_3d

# Step 3: Create a 3D Mesh
def create_mesh_from_point_cloud(points_3d):
    # Convert numpy array to Open3D Point Cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_3d)

    # Estimate normals for the point cloud
    point_cloud.estimate_normals()

    # Use Poisson surface reconstruction to generate a mesh
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9)[0]
    return mesh

# Step 4: Save the Mesh
def save_mesh(mesh, output_path):
    o3d.io.write_triangle_mesh(output_path, mesh)

# Main function
if __name__ == "__main__":
    # Load the depth map from a file
    depth_map_path = "/Users/janpoonthong/University/Year-3/Semester-1/selected_topic_in_computer_vision/thai_pottery/depth_map.png"
    depth_map = read_depth_map(depth_map_path)

    # Convert the depth map to a point cloud
    points_3d = depth_map_to_point_cloud(depth_map)

    # Generate a 3D mesh from the point cloud
    mesh = create_mesh_from_point_cloud(points_3d)

    # Save the 3D mesh to a file
    output_path = "/Users/janpoonthong/University/Year-3/Semester-1/selected_topic_in_computer_vision/thai_pottery/output_model.obj"
    save_mesh(mesh, output_path)

    # Visualize the mesh
    o3d.visualization.draw_geometries([mesh])
