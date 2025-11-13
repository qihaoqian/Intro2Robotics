import numpy as np
import cv2
import matplotlib.pyplot as plt
from read_data import load_vicon_dataset, load_cam_dataset, load_imu_dataset

def find_closest_index(t, ts_array):
    """
    Find the index of the closest timestamp in ts_array to t
    """
    idx = np.argmin(np.abs(ts_array - t))
    return idx

def build_panorama(cam_data, vicon_data):
    """
    Build a panorama image based on spherical and cylindrical projections
    """
    # Camera intrinsic parameters (example values, please fill in actual values)
    fx = 200
    fy = 200
    cx = 160
    cy = 120
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ], dtype=np.float64)
    K_inv = np.linalg.inv(K)

    # Get images and timestamps
    imgs = cam_data['cam']   # shape = (240, 320, 3, N)
    cam_ts = cam_data['ts'].flatten()  # shape = (N,)
    N = imgs.shape[3]

    # Get VICON rotation matrices and timestamps
    vicon_rots = vicon_data['rots']  # shape = (3, 3, N)
    vicon_ts = vicon_data['ts'].flatten()  # shape = (N,)

    # Determine the base frame (e.g. take the 0th frame as the base)
    base_img = imgs[..., 0]  # shape = (240, 320, 3)
    base_t = cam_ts[0]
    idx_base = find_closest_index(base_t, vicon_ts)
    R_base = vicon_rots[..., idx_base]  # 3x3

    h_img, w_img = base_img.shape[:2]

    # Define the horizontal and vertical field of view (in radians)
    h_fov = np.radians(60)
    v_fov = np.radians(45)

    # Precompute the spherical coordinates for each pixel
    # lon: horizontal angle range, from -h_fov/2 to +h_fov/2
    # lat: vertical angle range, from -v_fov/2 to +v_fov/2
    lon = np.linspace(-h_fov / 2, h_fov / 2, w_img)
    lat = np.linspace(-v_fov / 2, v_fov / 2, h_img)
    lon_grid, lat_grid = np.meshgrid(lon, lat)  # shape both (h_img, w_img)

    # Precompute the Cartesian coordinates for each pixel (r = 1)
    # Conversion formula: x = cos(lat) * cos(lon),  y = cos(lat) * sin(lon),  z = sin(lat)
    # Here r is in pixel units, representing the distance from the unit sphere to the origin
    r = 1
    cart_coords = np.empty((h_img, w_img, 3), dtype=np.float64)
    cart_coords[..., 0] = np.cos(lat_grid) * np.cos(lon_grid) * r
    cart_coords[..., 1] = np.cos(lat_grid) * np.sin(lon_grid) * r
    cart_coords[..., 2] = np.sin(lat_grid) * r

    # Prepare the panorama image
    panorama_w = int(2 * np.pi * 100)  # ~ 628
    panorama_h = int(np.pi * 100)      # ~ 314
    panorama = np.zeros((panorama_h, panorama_w, 3), dtype=np.uint8)

    from tqdm import tqdm
    for i in tqdm(range(N), desc="Processing images"):
        img_i = imgs[..., i]
        t_i = cam_ts[i]
        idx_i = find_closest_index(t_i, vicon_ts)
        R_i = vicon_rots[..., idx_i]  # 3x3
        R_rel_i = R_base.T @ R_i  # 3x3

        # Transform the Cartesian coordinates by the rotation matrix R_i
        world_coords = np.einsum('ij,hwj->hwi', R_rel_i, cart_coords)

        # Convert the world coordinates to spherical coordinates
        # Formula: lam = arctan2(y, x) ; phi = arcsin(z / r) ; r = sqrt(x^2 + y^2 + z^2)
        r = np.linalg.norm(world_coords, axis=-1)
        phi = np.arcsin(world_coords[..., 2] / r)
        lam = np.arctan2(world_coords[..., 1], world_coords[..., 0])

        # Map the spherical coordinates to the panorama image
        # Formula: cyl_x = int((lam + π) * 100) ; cyl_y = int((phi + π/2) * 100)
        cyl_x = ((lam + np.pi) * 100).astype(np.int32)
        cyl_y = ((phi + np.pi/2) * 100).astype(np.int32)

        # Select the valid pixels that are within the panorama image
        valid = (cyl_x >= 0) & (cyl_x < panorama_w) & (cyl_y >= 0) & (cyl_y < panorama_h)
        # Update the panorama image with the pixels from img_i
        panorama[cyl_y[valid], cyl_x[valid]] = img_i[valid]

    return panorama

def main(dataset_number=1):
    # Read data (example)
    cam_data = load_cam_dataset(dataset_number)
    vicon_data = load_vicon_dataset(dataset_number)

    # Build panorama
    panorama = build_panorama(cam_data, vicon_data)

    # Visualize
    plt.figure(figsize=(12, 6))
    plt.imshow(panorama)
    plt.axis('off')
    plt.gca().set_position([0, 0, 1, 1])  # Remove the white border
    plt.savefig(f"panorama_{dataset_number}.png", bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    for i in [1,2,8,9,10,11]:
        main(i)

