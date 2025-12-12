"""
Simulation environment setup - Create and configure PyBullet physics environment
"""

import numpy as np
import pybullet as p
import pybullet_data
import os
from utils.config import (
    TIME_STEP,
    GRAVITY,
    BOX_LENGTH,
    BOX_WIDTH,
    BOX_DEPTH,
    OBJECT_ORIENTATION,
    NUM_TARGETS,
    TARGET_AREA_X,
    TARGET_AREA_Y,
    TARGET_HEIGHT,
    MIN_TARGET_DISTANCE,
    TARGET_COLORS,
)


def setup_physics_environment(gui_mode=True):
    """
    Setup PyBullet physics environment

    Args:
        gui_mode: Whether to use GUI mode (True) or DIRECT mode (False)

    Returns:
        physicsClient: Physics client ID
    """
    if gui_mode:
        physicsClient = p.connect(p.GUI)
    else:
        physicsClient = p.connect(p.DIRECT)

    p.resetSimulation()
    p.setTimeStep(TIME_STEP)
    p.setGravity(*GRAVITY)

    # Set the path to URDF files bundled with PyBullet
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Load the plane URDF
    p.loadURDF("plane.urdf")

    # Reset debug visualizer camera position so we can see the robot up close
    p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=30, cameraPitch=-52, cameraTargetPosition=[0, 0, 0.5])

    return physicsClient


def create_box_obstacle(center, orientation=None, color=None, size=None):
    """
    Create a box obstacle in the simulation environment

    Args:
        center: Box center position
        orientation: Box orientation (Euler angles, uses default if None)
        color: Box color (RGBA, uses default if None)
        size: Box dimensions [length, width, depth] (uses default if None)

    Returns:
        boxId: PyBullet ID of the box
    """
    if orientation is None:
        orientation = OBJECT_ORIENTATION
    if color is None:
        color = [0.8, 0.0, 0.0, 1]
    if size is None:
        size = [BOX_LENGTH, BOX_WIDTH, BOX_DEPTH]

    geomBox = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size[0] / 2, size[1] / 2, size[2] / 2])
    visualBox = p.createVisualShape(p.GEOM_BOX, halfExtents=[size[0] / 2, size[1] / 2, size[2] / 2], rgbaColor=color)
    boxId = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=geomBox,
        baseVisualShapeIndex=visualBox,
        basePosition=np.array(center),
        baseOrientation=p.getQuaternionFromEuler(orientation),
    )

    return boxId


def create_multiple_random_targets(num_targets=None, seed=None):
    """
    Randomly create multiple targets in the planar environment

    Args:
        num_targets: Number of targets to create (uses config default if None)
        seed: Random seed (for reproducibility)

    Returns:
        target_ids: List of target IDs
        target_positions: List of target positions
    """
    if num_targets is None:
        num_targets = NUM_TARGETS

    if seed is not None:
        np.random.seed(seed)

    target_ids = []
    target_positions = []

    attempts = 0
    max_attempts = num_targets * 100  # Prevent infinite loop

    while len(target_positions) < num_targets and attempts < max_attempts:
        attempts += 1

        # Randomly generate position
        x = np.random.uniform(TARGET_AREA_X[0], TARGET_AREA_X[1])
        y = np.random.uniform(TARGET_AREA_Y[0], TARGET_AREA_Y[1])
        z = TARGET_HEIGHT
        new_position = np.array([x, y, z])

        # Check distance to existing targets
        too_close = False
        for existing_pos in target_positions:
            distance = np.linalg.norm(new_position[:2] - existing_pos[:2])
            if distance < MIN_TARGET_DISTANCE:
                too_close = True
                break

        if not too_close:
            # Select color
            color_idx = len(target_positions) % len(TARGET_COLORS)
            color = TARGET_COLORS[color_idx]

            # Create target
            target_id = create_box_obstacle(center=new_position, color=color)
            target_ids.append(target_id)
            target_positions.append(new_position)

            print(f"Created target {len(target_positions)}: position={new_position}, color={color[:3]}")

    if len(target_positions) < num_targets:
        print(f"Warning: Only created {len(target_positions)} targets (requested {num_targets})")

    return target_ids, target_positions


def load_panda_robot(initial_joint_position):
    """
    Load Franka Panda robot

    Args:
        initial_joint_position: List of initial joint positions

    Returns:
        pandaUid: PyBullet ID of the robot
    """
    pandaUid = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "franka_panda/panda.urdf"), useFixedBase=True)
    p.resetBasePositionAndOrientation(pandaUid, [0, 0, 0], [0, 0, 0, 1])

    return pandaUid
