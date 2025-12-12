"""
Target detection and selection module - For target identification and selection in multi-target scenarios
"""

import numpy as np
import pybullet as p
import cv2
from utils.camera_utils import get_camera_img_float, opengl_plot_world_to_pixelspace


def detect_targets_in_image(rgb_image, depth_image):
    """
    Detect targets in image using color segmentation

    Args:
        rgb_image: RGB image
        depth_image: Depth image

    Returns:
        detected_targets: List of detected targets, each element is dict {'pixel': [u, v], 'depth': z, 'color': [r, g, b]}
    """
    detected_targets = []

    # Convert to HSV for color segmentation
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # Define color range (for detecting colored targets)
    # Using saturation and brightness to filter colored objects
    lower_bound = np.array([0, 100, 100])
    upper_bound = np.array([180, 255, 255])

    # Create mask
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Morphological operations for noise removal
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Filter small contours
        area = cv2.contourArea(contour)
        if area < 100:  # Minimum area threshold
            continue

        # Calculate contour center
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Get depth
        if 0 <= cy < depth_image.shape[0] and 0 <= cx < depth_image.shape[1]:
            depth = depth_image[cy, cx]
            color = rgb_image[cy, cx]

            detected_targets.append({"pixel": [cx, cy], "depth": depth, "color": color.tolist(), "area": area})

    return detected_targets


def get_target_world_positions(target_ids):
    """
    Get positions of all targets in world coordinates

    Args:
        target_ids: List of PyBullet IDs for target objects

    Returns:
        positions: List of target positions, each element is [x, y, z]
    """
    positions = []
    for target_id in target_ids:
        pos, _ = p.getBasePositionAndOrientation(target_id)
        positions.append(np.array(pos))
    return positions


def select_closest_target(robot_base_position, target_positions, target_ids):
    """
    Select the target closest to the robot base

    Args:
        robot_base_position: Robot base position [x, y, z]
        target_positions: List of target positions
        target_ids: List of target IDs

    Returns:
        closest_target_id: ID of the closest target
        closest_target_position: Position of the closest target
        closest_distance: Closest distance
    """
    if len(target_positions) == 0:
        return None, None, None

    # Calculate distance from all targets to robot base (XY plane only)
    distances = []
    robot_base_xy = np.array(robot_base_position[:2])

    for target_pos in target_positions:
        target_xy = np.array(target_pos[:2])
        distance = np.linalg.norm(target_xy - robot_base_xy)
        distances.append(distance)

    # Find the closest target
    min_idx = np.argmin(distances)
    closest_target_id = target_ids[min_idx]
    closest_target_position = target_positions[min_idx]
    closest_distance = distances[min_idx]

    return closest_target_id, closest_target_position, closest_distance


def visualize_detected_targets(rgb_image, detected_targets, selected_pixel=None):
    """
    Visualize detected targets on image

    Args:
        rgb_image: RGB image
        detected_targets: List of detected targets
        selected_pixel: Pixel coordinates of selected target (optional)

    Returns:
        vis_image: Visualization image
    """
    vis_image = rgb_image.copy()

    # Draw all detected targets
    for target in detected_targets:
        u, v = target["pixel"]
        # Draw circle
        cv2.circle(vis_image, (u, v), 10, (0, 255, 0), 2)
        # Draw crosshair
        cv2.drawMarker(vis_image, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

    # Draw selected target (red)
    if selected_pixel is not None:
        u, v = selected_pixel
        cv2.circle(vis_image, (int(u), int(v)), 15, (255, 0, 0), 3)
        cv2.drawMarker(vis_image, (int(u), int(v)), (255, 0, 0), cv2.MARKER_CROSS, 30, 3)

    # Draw image center (blue)
    h, w = rgb_image.shape[:2]
    center = (w // 2, h // 2)
    cv2.circle(vis_image, center, 5, (0, 0, 255), -1)
    cv2.drawMarker(vis_image, center, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

    return vis_image


def find_target_in_image(target_world_position, camera_pos, camera_orn, viewMat, projMat, img_width, img_height):
    """
    Project target from world coordinates to image and get its pixel coordinates

    Args:
        target_world_position: Target position in world coordinates
        camera_pos: Camera position
        camera_orn: Camera orientation
        viewMat: View matrix
        projMat: Projection matrix
        img_width: Image width
        img_height: Image height

    Returns:
        pixel_coords: Pixel coordinates [u, v], returns None if target is not in view
    """
    pixel_coords = opengl_plot_world_to_pixelspace(target_world_position, viewMat, projMat, img_width, img_height)

    # Check if within image bounds
    u, v = pixel_coords
    if 0 <= u < img_width and 0 <= v < img_height:
        return pixel_coords
    else:
        return None
