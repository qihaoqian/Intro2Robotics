"""
Configuration file - Contains all global parameters
"""

import numpy as np

# Camera parameters (do not modify these settings)
CAMERA_WIDTH = 512  # Image width
CAMERA_HEIGHT = 512  # Image height
CAMERA_FOV = 120  # Camera field of view
CAMERA_FOCAL_DEPTH = 0.5 * CAMERA_HEIGHT / np.tan(0.5 * np.pi / 180 * CAMERA_FOV)
# Focal depth (pixel space)
CAMERA_ASPECT = CAMERA_WIDTH / CAMERA_HEIGHT  # Aspect ratio
CAMERA_NEAR = 0.02  # Near clipping plane (meters), do not set to zero
CAMERA_FAR = 100  # Far clipping plane (meters)

# Control targets (adjust these values if needed)
OBJECT_LOCATION_DESIRED = np.array([CAMERA_WIDTH / 2, CAMERA_HEIGHT / 2])
DESIRED_DEPTH = 0.5

# Visual servoing control gains (for unified IBVS/PBVS tuning)
PBVS_CONTROL_GAIN_TRANSLATION = 0.1
PBVS_CONTROL_GAIN_ROTATION = 0.02
IBVS_CONTROL_GAIN_TRANSLATION = 0.05
IBVS_CONTROL_GAIN_ROTATION = 0.01

# Physics simulation parameters
TIME_STEP = 0.001  # Simulation time step
GRAVITY = [0, 0, -9.8]  # Gravitational acceleration

# Object parameters (default parameters for single target)
BOX_LENGTH = 0.15
BOX_WIDTH = 0.15
BOX_DEPTH = 0.02
OBJECT_ORIENTATION = [0, 0, 0]

# Multi-target scene configuration
NUM_TARGETS = 5  # Number of targets
TARGET_AREA_X = (0.2, 0.8)  # Target placement area X range (meters)
TARGET_AREA_Y = (0.3, 1.2)  # Target placement area Y range (meters)
TARGET_HEIGHT = 0.01  # Target height (Z coordinate)
MIN_TARGET_DISTANCE = 0.2  # Minimum distance between targets (meters)

# Target color list (different colors for each target for distinction)
TARGET_COLORS = [
    [0.8, 0.0, 0.0, 1],  # Red
    [0.0, 0.8, 0.0, 1],  # Green
    [0.0, 0.0, 0.8, 1],  # Blue
    [0.8, 0.8, 0.0, 1],  # Yellow
    [0.8, 0.0, 0.8, 1],  # Magenta
    [0.0, 0.8, 0.8, 1],  # Cyan
    [0.8, 0.5, 0.0, 1],  # Orange
    [0.5, 0.0, 0.8, 1],  # Purple
]

# Robot initial joint position
INITIAL_JOINT_POSITION = [0, -np.pi / 4, np.pi / 4, -np.pi / 4, np.pi / 4, np.pi / 4, np.pi / 4, 0, 0, 0, 0, 0]

# Camera offset
CAMERA_OFFSET = 0.1  # Z-direction offset to avoid gripper

# Control parameters
MAX_ITERATIONS = 100  # Maximum iterations per target
CONVERGENCE_THRESHOLD = 3  # Convergence threshold (pixels)
