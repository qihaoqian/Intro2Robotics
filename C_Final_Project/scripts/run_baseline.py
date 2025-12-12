import numpy as np
import pybullet as p
import cv2
import argparse

from utils.config import (
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_FOCAL_DEPTH,
    OBJECT_LOCATION_DESIRED,
    INITIAL_JOINT_POSITION,
    NUM_TARGETS,
    MAX_ITERATIONS,
    CONVERGENCE_THRESHOLD,
)
from utils.simulation import setup_physics_environment, create_multiple_random_targets, load_panda_robot
from utils.camera_utils import get_camera_img_float, get_camera_view_and_projection_opencv
from utils.visualization import draw_coordinate_frame
from utils.controllers import get_image_jacobian, find_camera_control, find_joint_control
from utils.robot import EyeInHandRobot
from utils.target_selection import (
    get_target_world_positions,
    select_closest_target,
    find_target_in_image,
    visualize_detected_targets,
)

REWARD_WEIGHTS = {
    "tracking": 1.0,
    "energy": 100.0,
    "step_penalty": 10.0,
    "failure": 1000.0,
}


def compute_reward(error, joint_velocity, truncated):
    reward_w = REWARD_WEIGHTS
    energy_cost = np.sum(joint_velocity**2)
    tracking_reward = -reward_w["tracking"] * error
    energy_reward = -reward_w["energy"] * energy_cost
    step_penalty = -reward_w["step_penalty"]
    failure_reward = -reward_w["failure"] * truncated
    reward = tracking_reward + energy_reward + step_penalty + failure_reward

    reward_info = {
        "tracking_reward": tracking_reward,
        "energy_reward": energy_reward,
        "step_penalty": step_penalty,
        "failure_reward": failure_reward,
    }
    return reward, reward_info


def run_autonomous_visual_servoing(num_targets=NUM_TARGETS, seed=None, enable_video=False, gui=False):
    print("=" * 80)
    print("Autonomous Multi-Target Visual Servoing System")
    print("=" * 80)

    print("\n[1/5] Initializing physics environment...")
    physicsClient = setup_physics_environment(gui_mode=gui)

    print(f"\n[2/5] Creating {num_targets} random targets...")
    target_ids, target_positions = create_multiple_random_targets(num_targets=num_targets, seed=seed)
    print(f"Successfully created {len(target_ids)} targets")

    # Create robot instance
    print("\n[3/5] Loading robot...")
    pandaUid = load_panda_robot(INITIAL_JOINT_POSITION)
    robot = EyeInHandRobot(pandaUid, INITIAL_JOINT_POSITION)
    p.stepSimulation()  # Initialize robot

    # Get robot base position
    robot_base_pos, _ = p.getBasePositionAndOrientation(pandaUid)
    print(f"Robot base position: {robot_base_pos}")

    # Select closest target
    print("\n[4/5] Selecting closest target...")
    target_positions_list = get_target_world_positions(target_ids)
    closest_target_id, closest_target_pos, closest_distance = select_closest_target(
        robot_base_pos, target_positions_list, target_ids
    )
    print(f"Closest target: ID={closest_target_id}, position={closest_target_pos}, distance={closest_distance:.3f}m")

    # Start visual servoing
    print("\n[5/5] Starting visual servoing control...")
    if enable_video:
        log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "autonomous_visual_servoing.mp4")

    camera_frameId = []
    total_reward = 0.0
    total_tracking_reward = 0.0
    total_energy_reward = 0.0
    total_step_penalty = 0.0
    total_failure_reward = 0.0
    error = float("nan")
    mission_succeeded = False
    truncated = False

    for iteration in range(MAX_ITERATIONS):
        # 获取相机位姿
        cameraPosition, cameraOrientation = robot.get_ee_position()
        rgb, depth = get_camera_img_float(cameraPosition, cameraOrientation)
        view_mat, proj_mat = get_camera_view_and_projection_opencv(cameraPosition, cameraOrientation)
        target_pixel = find_target_in_image(
            closest_target_pos, cameraPosition, cameraOrientation, view_mat, proj_mat, CAMERA_WIDTH, CAMERA_HEIGHT
        )
        image_jacobian = get_image_jacobian(
            target_pixel[0],
            target_pixel[1],
            depth,
            CAMERA_FOCAL_DEPTH,
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
            cameraOrientation,
        )
        delta_X, delta_Omega = find_camera_control(OBJECT_LOCATION_DESIRED, target_pixel, image_jacobian)
        new_joint_positions, joint_velocities = find_joint_control(robot, delta_X, delta_Omega)
        # Update robot state
        robot.set_joint_position(new_joint_positions)
        p.stepSimulation()

        # Draw camera coordinate frame
        camera_frameId = draw_coordinate_frame(cameraPosition, cameraOrientation, length=0.15, frameId=camera_frameId)

        # Get next frame info and compute reward
        cameraPosition_next, cameraOrientation_next = robot.get_ee_position()
        rgb_next, depth_next = get_camera_img_float(cameraPosition_next, cameraOrientation_next)
        view_mat_next, proj_mat_next = get_camera_view_and_projection_opencv(
            cameraPosition_next, cameraOrientation_next
        )
        target_pixel_next = find_target_in_image(
            closest_target_pos,
            cameraPosition_next,
            cameraOrientation_next,
            view_mat_next,
            proj_mat_next,
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
        )

        # Reward calculation
        if target_pixel_next is None:
            print(f"✗ Target lost, mission failed (iteration {iteration}), total_reward={total_reward:.3f}")
            truncated = True
            error = np.linalg.norm(np.array([CAMERA_WIDTH / 2, CAMERA_HEIGHT / 2]))
        else:
            # Calculate error
            error = np.linalg.norm(np.array(OBJECT_LOCATION_DESIRED) - np.array(target_pixel_next))
        reward, reward_info = compute_reward(error, joint_velocities, truncated)
        total_reward += reward
        total_tracking_reward += reward_info["tracking_reward"]
        total_energy_reward += reward_info["energy_reward"]
        total_step_penalty += reward_info["step_penalty"]
        total_failure_reward += reward_info["failure_reward"]
        # Print status
        if iteration % 1 == 0:
            print(
                f"迭代 {iteration}: "
                f"reward={reward:.3f}, tracking={reward_info['tracking_reward']:.3f}, energy={reward_info['energy_reward']:.6f}, step_penalty={reward_info['step_penalty']:.3f}, failure={reward_info['failure_reward']:.3f}"
            )
        if truncated:
            break
        # Check convergence
        if error < CONVERGENCE_THRESHOLD:
            print(f"\n✓ Target converged to image center! (iteration {iteration}, error={error:.2f}px)")
            mission_succeeded = True
            break

    if enable_video:
        p.stopStateLogging(log_id)

    # Summary
    print("\n" + "=" * 80)
    print("Visual servoing completed!")
    print(f"Total iterations: {iteration + 1}")
    print(f"Final error: {error:.2f}px")
    if truncated:
        print("Convergence status: ✗ Target lost (failed)")
    else:
        print(f"Convergence status: {'✓ Converged' if mission_succeeded else '✗ Not converged'}")
    print(f"Cumulative reward: {total_reward:.3f}")
    print("=" * 80)

    # Cleanup
    cv2.destroyAllWindows()
    p.disconnect()

    return (
        total_reward,
        total_tracking_reward,
        total_energy_reward,
        total_step_penalty,
        total_failure_reward,
        mission_succeeded,
    )


def main():
    parser = argparse.ArgumentParser(description="Run baseline servoing N times and compute average reward")
    parser.add_argument("-n", type=int, required=True, help="Number of runs")
    parser.add_argument("--video", action="store_true", help="Enable video recording")
    parser.add_argument(
        "--gui",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to display PyBullet GUI window (default: off, use --gui to enable)",
    )
    args = parser.parse_args()

    total_rewards = []
    total_tracking_rewards = []
    total_energy_rewards = []
    total_step_penalties = []
    total_failure_rewards = []
    total_mission_succeeded = []
    for i in range(args.n):
        print(f"\n------ Run {i+1} ------")
        (
            total_reward,
            total_tracking_reward,
            total_energy_reward,
            total_step_penalty,
            total_failure_reward,
            mission_succeeded,
        ) = run_autonomous_visual_servoing(seed=None, enable_video=False, gui=args.gui)
        total_rewards.append(total_reward)
        total_tracking_rewards.append(total_tracking_reward)
        total_energy_rewards.append(total_energy_reward)
        total_step_penalties.append(total_step_penalty)
        total_failure_rewards.append(total_failure_reward)

    print("\n" + "=" * 40)
    print(f"Completed {args.n} runs, average results:")
    print(f"Average success rate: {sum(total_mission_succeeded)/args.n:.3f}")
    print(f"Average cumulative reward: {sum(total_rewards)/args.n:.3f}")
    print(f"Average Tracking Reward: {sum(total_tracking_rewards)/args.n:.3f}")
    print(f"Average Energy Reward: {sum(total_energy_rewards)/args.n:.3f}")
    print(f"Average Step Penalty: {sum(total_step_penalties)/args.n:.3f}")
    print(f"Average Failure Reward: {sum(total_failure_rewards)/args.n:.3f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
