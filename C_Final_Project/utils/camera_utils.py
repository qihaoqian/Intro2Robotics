"""
Camera utility functions - Handle camera view, projection, and image acquisition
"""

import numpy as np
import pybullet as p
from utils.config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FOV, CAMERA_ASPECT, CAMERA_NEAR, CAMERA_FAR


def get_camera_view_and_projection_opencv(camera_pos, camera_orn):
    """
    Get camera view and projection matrices (OpenCV format)

    Args:
        camera_pos: Camera position (3D vector)
        camera_orn: Camera orientation (3x3 rotation matrix)

    Returns:
        viewMat: 4x4 view matrix
        projMat: 4x4 projection matrix
    """
    __camera_view_matrix_opengl = p.computeViewMatrix(
        cameraEyePosition=camera_pos,
        cameraTargetPosition=camera_pos + camera_orn[:, 2],
        cameraUpVector=-camera_orn[:, 1],
    )

    __camera_projection_matrix_opengl = p.computeProjectionMatrixFOV(CAMERA_FOV, CAMERA_ASPECT, CAMERA_NEAR, CAMERA_FAR)

    _, _, rgbImg, depthImg, _ = p.getCameraImage(
        CAMERA_WIDTH,
        CAMERA_HEIGHT,
        __camera_view_matrix_opengl,
        __camera_projection_matrix_opengl,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )

    # Return camera view and projection matrices suitable for OpenCV
    viewMat = np.array(__camera_view_matrix_opengl).reshape(4, 4).T
    projMat = np.array(__camera_projection_matrix_opengl).reshape(4, 4).T
    return viewMat, projMat


def get_camera_img_float(camera_pos, camera_orn):
    """
    Get image and depth map from camera at a given position and orientation in space

    Args:
        camera_pos: Camera position (3D vector)
        camera_orn: Camera orientation (3x3 rotation matrix)

    Returns:
        rgb_image: RGB image (uint8 format)
        depth_image: Linearized depth map (float32 format)
    """
    __camera_view_matrix_opengl = p.computeViewMatrix(
        cameraEyePosition=camera_pos,
        cameraTargetPosition=camera_pos + camera_orn[:, 2],
        cameraUpVector=-camera_orn[:, 1],
    )

    __camera_projection_matrix_opengl = p.computeProjectionMatrixFOV(CAMERA_FOV, CAMERA_ASPECT, CAMERA_NEAR, CAMERA_FAR)

    width, height, rgbImg, nonlinDepthImg, _ = p.getCameraImage(
        CAMERA_WIDTH,
        CAMERA_HEIGHT,
        __camera_view_matrix_opengl,
        __camera_projection_matrix_opengl,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )

    # Adjust clipping and non-linear distance, i.e. 1/d (0 is nearest=near, 1 is farthest=far)
    depthImgLinearized = (
        CAMERA_FAR * CAMERA_NEAR / (CAMERA_FAR + CAMERA_NEAR - (CAMERA_FAR - CAMERA_NEAR) * nonlinDepthImg)
    )

    # Convert to numpy arrays and RGB-D image
    rgb_image = np.array(rgbImg[:, :, :3], dtype=np.uint8)
    depth_image = np.array(depthImgLinearized, dtype=np.float32)
    return rgb_image, depth_image


def opengl_plot_world_to_pixelspace(pt_in_3D_to_project, viewMat, projMat, imgWidth, imgHeight):
    """
    Project 3D point from world coordinates to image pixel space
    Used for debugging, e.g. given a known position in world, verify it appears in camera

    Args:
        pt_in_3D_to_project: 3D point to project
        viewMat: View matrix
        projMat: Projection matrix
        imgWidth: Image width
        imgHeight: Image height

    Returns:
        [u, v]: Pixel coordinates
    """
    pt_in_3D_to_project = np.append(pt_in_3D_to_project, 1)

    # Transform to camera frame
    pt_in_3D_in_camera_frame = viewMat @ pt_in_3D_to_project

    # Transform coordinates to get normalized device coordinates (before scaling)
    uvzw = projMat @ pt_in_3D_in_camera_frame

    # Scale to get normalized device coordinates
    uvzw_NDC = uvzw / uvzw[3]

    # x, y specify the lower left corner of the viewport rectangle, in pixels. Initial value is (0,0)
    u = ((uvzw_NDC[0] + 1) / 2.0) * imgWidth
    v = ((1 - uvzw_NDC[1]) / 2.0) * imgHeight

    return [int(u), int(v)]
