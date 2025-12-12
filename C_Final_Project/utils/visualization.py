"""
Visualization utility functions - For drawing coordinate frames etc. in PyBullet
"""

import numpy as np
import pybullet as p


def draw_coordinate_frame(position, orientation, length, frameId=[]):
    """
    Draw an x, y, z coordinate frame in world coordinates with scaled axis lengths

    Args:
        position: 3-element numpy array, coordinate frame origin position
        orientation: 3x3 numpy matrix, coordinate frame orientation
        length: Length of the x, y, z axes to draw
        frameId: Unique ID of the coordinate frame. If provided, erases the previous frame position

    Returns:
        frameId: Tuple of frame IDs (for subsequent updates)
    """
    if len(frameId) != 0:
        p.removeUserDebugItem(frameId[0])
        p.removeUserDebugItem(frameId[1])
        p.removeUserDebugItem(frameId[2])

    # Draw x-axis (red)
    lineIdx = p.addUserDebugLine(position, position + np.dot(orientation, [length, 0, 0]), [1, 0, 0])
    # Draw y-axis (green)
    lineIdy = p.addUserDebugLine(position, position + np.dot(orientation, [0, length, 0]), [0, 1, 0])
    # Draw z-axis (blue)
    lineIdz = p.addUserDebugLine(position, position + np.dot(orientation, [0, 0, length]), [0, 0, 1])

    return lineIdx, lineIdy, lineIdz
