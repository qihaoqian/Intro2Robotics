"""
Controller module - Contains image Jacobian, camera control, and joint control algorithms
"""

import numpy as np
from utils.config import IBVS_CONTROL_GAIN_TRANSLATION, IBVS_CONTROL_GAIN_ROTATION


def get_image_jacobian(u_px, v_px, depth_img, f, img_width, img_height, camera_orientation):
    """
    Compute image Jacobian matrix

    Args:
        u_px: Pixel u coordinate
        v_px: Pixel v coordinate
        depth_img: Depth image
        f: Focal length
        img_width: Image width
        img_height: Image height
        camera_orientation: Camera orientation (3x3 rotation matrix)

    Returns:
        L_world: Image Jacobian matrix in world coordinates (2x6)
    """
    # Get depth value
    u_idx = int(np.clip(round(u_px), 0, depth_img.shape[1] - 1))
    v_idx = int(np.clip(round(v_px), 0, depth_img.shape[0] - 1))
    Z = float(depth_img[v_idx, u_idx])
    if Z < 0.01:
        Z = 0.01

    # print("u_idx, v_idx, Z", u_idx, v_idx, Z)

    # Normalized image coordinates
    x = (u_px - img_width / 2.0) / f
    y = (v_px - img_height / 2.0) / f

    # Image Jacobian on normalized plane
    L_norm = np.array(
        [[-1.0 / Z, 0.0, x / Z, x * y, -(1.0 + x * x), y], [0.0, -1.0 / Z, y / Z, 1.0 + y * y, -x * y, -x]], dtype=float
    )

    # Convert to pixel space
    L_cam = np.empty_like(L_norm)
    L_cam[0, :] = f * L_norm[0, :]
    L_cam[1, :] = f * L_norm[1, :]

    # Transform from camera frame to world frame
    R_wc = np.asarray(camera_orientation, dtype=float)
    R_cw = R_wc.T
    T = np.block([[R_cw, np.zeros((3, 3))], [np.zeros((3, 3)), R_cw]])

    L_world = L_cam @ T

    return L_world


def find_camera_control(object_loc_des, object_loc, image_jacobian):
    """
    Compute camera control commands

    Args:
        object_loc_des: Desired object position (pixel coordinates)
        object_loc: Current object position (pixel coordinates)
        image_jacobian: Image Jacobian matrix (2x6)

    Returns:
        delta_X: Translational velocity (3D vector)
        delta_Omega: Rotational velocity (3D vector)
    """
    error = np.array(object_loc_des) - np.array(object_loc)
    # print(f"ibvserror: {np.linalg.norm(error)}")

    # Optional: Use translation only
    # image_jacobian[:, 3:6] = 0

    # Optional: Use rotation only
    # image_jacobian[:, 0:3] = 0

    # Compute pseudo-inverse
    J_pseudo_inv = np.linalg.pinv(image_jacobian)

    # Compute camera velocity
    camera_velocity = J_pseudo_inv @ error

    velocity_trans = camera_velocity[0:3]
    velocity_rot = camera_velocity[3:6]

    delta_X = IBVS_CONTROL_GAIN_TRANSLATION * velocity_trans
    delta_Omega = IBVS_CONTROL_GAIN_ROTATION * velocity_rot

    return delta_X, delta_Omega


def find_joint_control(robot, delta_X, delta_Omega):
    """
    Compute joint control commands

    Args:
        robot: Robot instance
        delta_X: Desired translational velocity (3D vector)
        delta_Omega: Desired rotational velocity (3D vector)

    Returns:
        new_joint_positions: New joint positions
    """
    current_joint_angles = robot.get_current_joint_angles()

    # Get robot Jacobian
    J_robot = robot.get_jacobian_at_current_position()

    # Combine camera velocity
    camera_velocity = np.hstack((delta_X, delta_Omega))

    # Compute joint velocity
    joint_velocity = np.linalg.pinv(J_robot) @ camera_velocity

    # Update joint positions
    new_joint_positions = current_joint_angles + joint_velocity

    return new_joint_positions, joint_velocity


def find_pbvs_camera_control(
    target_pos_world,
    camera_position,
    camera_orientation,
    desired_pixel,
    desired_depth,
    img_width,
    img_height,
    focal_depth,
    k_trans,
    k_rot,
):
    """
    PBVS control: Compute camera velocity based on target world coordinates and current camera pose

    Args:
        target_pos_world: Target position in world coordinates (3,)
        camera_position: Camera position in world coordinates (3,)
        camera_orientation: Camera orientation as 3x3 rotation matrix (world frame)
        target_pixel: Target position in image (u, v)
        desired_pixel: Desired image coordinates (u, v)
        desired_depth: Desired target depth (camera frame, meters)
        img_width: Image width
        img_height: Image height
        focal_depth: Focal length (pixels)
        k_trans: Translation control gain
        k_rot: Rotation control gain

    Returns:
        delta_X: Translational velocity (3,)
        delta_Omega: Angular velocity (3,)
    """
    R_wc = np.asarray(camera_orientation, dtype=float)
    t_wc = np.asarray(camera_position, dtype=float)
    target_pos_world = np.asarray(target_pos_world, dtype=float)
    # print(f"camera_position: {camera_position}, target_pos_world: {target_pos_world}")

    # Target position in camera frame
    p_c = R_wc.T @ (target_pos_world - t_wc)

    # Desired target position derived from desired pixel and desired depth (camera frame)
    x_des = (desired_pixel[0] - img_width / 2.0) / focal_depth * desired_depth
    y_des = (desired_pixel[1] - img_height / 2.0) / focal_depth * desired_depth
    p_des = np.array([x_des, y_des, desired_depth], dtype=float)

    # Translation error (camera frame)
    trans_error = p_c - p_des
    print(f"pbvs_trans_error: {np.linalg.norm(trans_error)}")
    delta_X = k_trans * trans_error

    # Rotation error: Align camera optical axis with target direction
    z_cam = np.array([0.0, 0.0, 1.0], dtype=float)
    dir_to_target = p_c / (np.linalg.norm(p_c) + 1e-9)
    rot_axis = np.cross(z_cam, dir_to_target)
    rot_axis_norm = np.linalg.norm(rot_axis)
    if rot_axis_norm < 1e-9:
        delta_Omega = np.zeros(3)
    else:
        rot_axis_unit = rot_axis / rot_axis_norm
        angle_err = np.arcsin(np.clip(rot_axis_norm, -1.0, 1.0))
        delta_Omega = k_rot * angle_err * rot_axis_unit

    delta_X = R_wc @ delta_X
    delta_Omega = R_wc @ delta_Omega

    return delta_X, delta_Omega


def compute_roll_twist_sec(camera_orientation, k_roll=1.0):
    """
    Compute secondary roll twist for camera to maintain zero roll relative to world z-axis

    Args:
        camera_orientation: 3x3 rotation matrix representing camera orientation in world frame
        k_roll: Gain parameter for roll correction, default 1.0

    Returns:
        psi: Roll angle of camera around optical axis (z) relative to world z-axis (radians)
        v_sec_world: 6D twist vector in world frame representing desired secondary control to minimize roll
                     First 3 elements are zero (no translation), last 3 elements are angular velocity
    """
    R_wc = np.asarray(camera_orientation, dtype=float)
    xw, yw, zw = R_wc[:, 0], R_wc[:, 1], R_wc[:, 2]
    g = np.array([0.0, 0.0, 1.0])
    g_perp = g - np.dot(g, zw) * zw
    nrm = np.linalg.norm(g_perp)
    if nrm < 1e-9:
        return 0.0, np.zeros(6)
    g_perp /= nrm

    psi = np.arctan2(np.dot(xw, g_perp), np.dot(yw, g_perp))
    omega_sec = -k_roll * psi * zw
    v_sec_world = np.zeros(6)
    v_sec_world[3:] = omega_sec
    return psi, v_sec_world


# def find_pbvs_camera_control(target_position, camera_position, camera_orientation, desired_pixel, desired_depth=0.5, focal_length=CAMERA_FOCAL_DEPTH):
