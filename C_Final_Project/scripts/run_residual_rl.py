import argparse
from curses import noecho
from typing import Dict, Optional, Tuple

import gymnasium as gym
from gymnasium.wrappers import RescaleAction
import numpy as np
import pybullet as p
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from utils.controllers import get_image_jacobian, find_camera_control, find_joint_control
from utils.camera_utils import get_camera_img_float, get_camera_view_and_projection_opencv
from utils.config import (
    CAMERA_FOCAL_DEPTH,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CONVERGENCE_THRESHOLD,
    INITIAL_JOINT_POSITION,
    MAX_ITERATIONS,
    OBJECT_LOCATION_DESIRED,
    NUM_TARGETS,
)
from utils.robot import EyeInHandRobot
from utils.simulation import create_multiple_random_targets, setup_physics_environment, load_panda_robot
from utils.target_selection import find_target_in_image, select_closest_target, get_target_world_positions


class ResidualVisualServoEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        gui: bool = False,
        max_steps: int = 100,
        reward_weights: Optional[Dict[str, float]] = None,
        seed: int = None,
    ):
        super().__init__()
        self.gui = gui
        self.max_steps = max_steps
        self.reward_w = reward_weights or {
            "tracking": 1.0,
            "energy": 10.0,
            "failure": 1000.0,
            "step_penalty": 20.0,
        }
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

        self.physics_client = None
        self.robot: Optional[EyeInHandRobot] = None
        self.target_positions = None
        self.current_target_idx = None
        self.target_ids = None
        self.step_count = 0

        self._reset_scene()
        self.action_dim = 6
        # target_pixel + depth + joint_angles + joint_velocity + delta_X + delta_Omega
        self.obs_dim = 2 + 1 + 7 + 7 + 3 + 3

        self.action_space = spaces.Box(low=-100.0, high=100.0, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

    def _compute_ibvs_suggestion(self, target_pixel, depth_img, camera_orn):
        if target_pixel is None:
            return np.zeros(3), np.zeros(3)

        image_jacobian = get_image_jacobian(
            target_pixel[0], target_pixel[1], depth_img, CAMERA_FOCAL_DEPTH, CAMERA_WIDTH, CAMERA_HEIGHT, camera_orn
        )
        delta_X, delta_Omega = find_camera_control(OBJECT_LOCATION_DESIRED, target_pixel, image_jacobian)
        return delta_X, delta_Omega

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._seed = seed
            self.np_random = np.random.default_rng(seed)
        self.step_count = 0
        self._reset_scene()

        target_pixel, depth_img, camera_pos, camera_orn = self._capture_target()

        self.current_ibvs_X, self.current_ibvs_Omega = self._compute_ibvs_suggestion(
            target_pixel, depth_img, camera_orn
        )

        obs = self._build_obs(
            target_pixel,
            depth_img,
            self.robot.get_current_joint_angles(),
            np.zeros(7),
            self.current_ibvs_X,
            self.current_ibvs_Omega,
        )

        return obs, {}
    
    def step(self, action: np.ndarray):
        self.step_count += 1

        delta_X_nominal = self.current_ibvs_X
        delta_Omega_nominal = self.current_ibvs_Omega

        soft_action = np.tanh(action)

        residual_coeff = np.array(soft_action * 0.3 + 1.0, dtype=np.float32)

        delta_X_cmd = delta_X_nominal * residual_coeff[:3]
        delta_Omega_cmd = delta_Omega_nominal * residual_coeff[3:]

        joint_positions_next, joint_velocity = find_joint_control(self.robot, delta_X_cmd, delta_Omega_cmd)
        self.robot.set_joint_position(joint_positions_next)
        p.stepSimulation()

        target_pixel_next, depth_img_next, camera_pos_next, camera_orn_next = self._capture_target()

        next_ibvs_X, next_ibvs_Omega = self._compute_ibvs_suggestion(target_pixel_next, depth_img_next, camera_orn_next)
        _, joint_velocity_next = find_joint_control(self.robot, next_ibvs_X, next_ibvs_Omega)

        self.current_ibvs_X = next_ibvs_X
        self.current_ibvs_Omega = next_ibvs_Omega

        joint_angles_next = self.robot.get_current_joint_angles()

        obs = self._build_obs(
            target_pixel_next,
            depth_img_next,
            joint_angles_next,
            joint_velocity_next,
            next_ibvs_X,
            next_ibvs_Omega,
        )

        terminated, truncated = self._check_done(target_pixel_next)
        reward, reward_info = self._compute_reward(target_pixel_next, joint_velocity, truncated)

        print(
            f"Step: {self.step_count:03d},coeff: {np.array2string(residual_coeff, precision=2)}, Reward: {reward:7.4f}, reward_info: {reward_info}"
        )

        info = {"reward": reward, "reward_info": reward_info}

        return obs, reward, terminated, truncated, info

    def render(self):
        return None

    def close(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None

    def _reset_scene(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
        self.physics_client = setup_physics_environment(gui_mode=self.gui)
        seed_for_targets = int(self.np_random.integers(1_000_000_000))
        self.target_ids, self.target_positions = create_multiple_random_targets(
            num_targets=NUM_TARGETS, seed=seed_for_targets
        )
        panda_uid = load_panda_robot(INITIAL_JOINT_POSITION)
        self.robot = EyeInHandRobot(panda_uid, INITIAL_JOINT_POSITION)
        robot_base_pos, _ = p.getBasePositionAndOrientation(panda_uid)
        closest_id, closest_pos, _ = select_closest_target(robot_base_pos, self.target_positions, self.target_ids)
        self.current_target_idx = self.target_ids.index(closest_id) if closest_id is not None else None

    def _capture_target(self) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        camera_pos, camera_orn = self.robot.get_ee_position()
        rgb, depth = get_camera_img_float(camera_pos, camera_orn)
        view_mat, proj_mat = get_camera_view_and_projection_opencv(camera_pos, camera_orn)
        if self.current_target_idx is None:
            return None, depth, camera_pos, camera_orn
        target_pixel = find_target_in_image(
            self.target_positions[self.current_target_idx],
            camera_pos,
            camera_orn,
            view_mat,
            proj_mat,
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
        )
        return target_pixel, depth, camera_pos, camera_orn

    def _build_obs(self, target_pixel, depth_img, joint_angles, joint_vel, delta_X, delta_Omega):
        if target_pixel is None:
            return np.zeros(self.obs_dim)
        else:
            vis_error = (target_pixel - OBJECT_LOCATION_DESIRED) / np.array([CAMERA_WIDTH / 2, CAMERA_HEIGHT / 2])

        depth_val = np.array([depth_img[int(target_pixel[1]), int(target_pixel[0])]], dtype=np.float32)

        norm_joint_angles = joint_angles / np.pi

        delta_X = np.array(delta_X, dtype=np.float32)
        delta_Omega = np.array(delta_Omega, dtype=np.float32)

        obs = np.concatenate(
            [
                vis_error.flatten(),
                depth_val.flatten(),
                norm_joint_angles.flatten(),
                joint_vel.flatten(),
                delta_X.flatten(),
                delta_Omega.flatten(),
            ]
        ).astype(np.float32)

        return obs

    def _get_obs(self):
        target_pixel, depth_img, _, _ = self._capture_target()
        joint_angles = self.robot.get_current_joint_angles()

        _, joint_velocity, _ = self.robot.get_active_joint_states()
        joint_velocity = np.array(joint_velocity[: self.robot.numActiveJoints], dtype=np.float32)
        return self._build_obs(target_pixel, depth_img, joint_angles, joint_velocity, np.zeros(3), np.zeros(3))

    def _compute_reward(self, target_pixel, joint_velocity, truncated):
        if target_pixel is None:
            tracking_cost = np.linalg.norm(np.array([CAMERA_WIDTH / 2, CAMERA_HEIGHT / 2]))
        else:
            tracking_cost = float(np.linalg.norm(np.array(target_pixel) - OBJECT_LOCATION_DESIRED))

        energy_cost = np.sum(joint_velocity**2)
        tracking_reward = -self.reward_w["tracking"] * tracking_cost
        energy_reward = -self.reward_w["energy"] * energy_cost
        failure_reward = -self.reward_w["failure"] * truncated
        step_penalty = -self.reward_w["step_penalty"]
        reward = tracking_reward + energy_reward + failure_reward + step_penalty

        reward_info = {
            "tracking_reward": "{:.5f}".format(tracking_reward),
            "energy_reward": "{:.2e}".format(energy_reward),
            "failure_reward": float(failure_reward),
            "step_penalty": float(step_penalty),
        }
        return reward, reward_info

    def _check_done(self, target_pixel):
        if target_pixel is None:
            return False, True
        error = np.linalg.norm(np.array(target_pixel) - OBJECT_LOCATION_DESIRED)
        terminated = (self.step_count >= self.max_steps) or (error < CONVERGENCE_THRESHOLD)
        truncated = False
        return terminated, truncated


class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.current_episode_rewards = {}
        self.current_episode_total = 0.0
        self.current_episode_len = 0
        self.step_energy_reward = []
        self.step_tracking_reward = []
        self.step_penalty = []
        self.episode_idx = 0

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        done = self.locals["dones"][0]
        step_reward = self.locals["rewards"][0]

        self.current_episode_total += step_reward
        self.current_episode_len += 1

        reward_info = info.get("reward_info", {})
        energy_step = float(reward_info.get("energy_reward", 0.0))
        tracking_step = float(reward_info.get("tracking_reward", 0.0))
        step_penalty = float(reward_info.get("step_penalty", 0.0))
        self.step_energy_reward.append(energy_step)
        self.step_tracking_reward.append(tracking_step)
        self.step_penalty.append(step_penalty)
        if done:
            self.episode_idx += 1
            self.logger.record("reward_episode/total", self.current_episode_total)
            self.logger.record("reward_episode/episode_idx", self.episode_idx)

            self.logger.record("reward_episode/energy_reward_sum", sum(self.step_energy_reward))
            self.logger.record("reward_episode/tracking_reward_sum", sum(self.step_tracking_reward))
            self.logger.record("reward_episode/step_penalty_sum", sum(self.step_penalty))
            self.logger.record("reward_episode/length", self.current_episode_len)

            self.current_episode_rewards = {}
            self.current_episode_total = 0.0
            self.current_episode_len = 0
            self.step_energy_reward = []
            self.step_tracking_reward = []
            self.step_penalty = []
            self.logger.dump(step=self.episode_idx)

        return True


def make_env(gui: bool = False, seed: int = None):
    env = ResidualVisualServoEnv(gui=gui, seed=seed)
    return Monitor(env)


def train(
    total_timesteps: int = 20000,
    n_steps: int = 512,
    batchsize: int = 64,
    gui: bool = False,
    seed: int = None,
    model_load_path: str = None,
    model_save_path: str = "ppo_residual_servo",
    tensorboard_log: str = "runs/residual_rl",
):
    env = DummyVecEnv([lambda: make_env(gui=gui, seed=seed)])
    if model_load_path is not None:
        model = PPO.load(model_load_path, env=env)
        if model.n_steps != n_steps:
            model.n_steps = n_steps
            model.rollout_buffer = RolloutBuffer(
                buffer_size=n_steps,
                observation_space=model.observation_space,
                action_space=model.action_space,
                device=model.device,
                gamma=model.gamma,
                gae_lambda=model.gae_lambda,
                n_envs=model.n_envs,
            )
        model.batch_size = batchsize
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=seed,
            tensorboard_log=tensorboard_log,
            n_steps=n_steps,
            batch_size=batchsize,
        )
    callback = RewardLoggingCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback, tb_log_name="residual_rl", progress_bar=True)
    model.save(model_save_path)
    return model


def evaluate(model_load_path: str, episodes: int = 5, gui: bool = True, seed: int = None):
    env = make_env(gui=gui, seed=seed)
    model = PPO.load(model_load_path)

    total_rewards = []
    total_tracking_rewards = []
    total_energy_rewards = []
    total_step_penalties = []
    total_failure_rewards = []

    for ep in range(episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_tracking = 0.0
        ep_energy = 0.0
        ep_step_penalty = 0.0
        ep_failure = 0.0
        for _ in range(MAX_ITERATIONS):
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            reward_info = info.get("reward_info", {})
            ep_tracking += float(reward_info.get("tracking_reward", 0.0))
            ep_energy += float(reward_info.get("energy_reward", 0.0))
            ep_step_penalty += float(reward_info.get("step_penalty", 0.0))
            ep_failure += float(reward_info.get("failure_reward", 0.0))
            if terminated or truncated:
                break

        total_rewards.append(ep_reward)
        total_tracking_rewards.append(ep_tracking)
        total_energy_rewards.append(ep_energy)
        total_step_penalties.append(ep_step_penalty)
        total_failure_rewards.append(ep_failure)

        print(
            f"[Eval] Episode {ep + 1}: reward={ep_reward:.3f}, tracking={ep_tracking:.3f}, energy={ep_energy:.6f}, step_penalty={ep_step_penalty:.2f}, failure={ep_failure:.1f}"
        )

    env.close()
    print("\n" + "=" * 40)
    print(f"Evaluation results over {episodes} episodes:")
    print(f"Average cumulative reward: {np.mean(total_rewards):.3f}")
    print(f"Average Tracking Reward: {np.mean(total_tracking_rewards):.3f}")
    print(f"Average Energy Reward: {np.mean(total_energy_rewards):.3f}")
    print(f"Average Step Penalty: {np.mean(total_step_penalties):.3f}")
    print(f"Average Failure Reward: {np.mean(total_failure_rewards):.3f}")
    print("=" * 40)


def parse_args():
    parser = argparse.ArgumentParser(description="Residual PPO for visual servoing")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--eval", action="store_true", help="Run evaluation")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes for evaluation")
    parser.add_argument("--n-steps", type=int, default=512, help="Number of samples per training step")
    parser.add_argument("--batchsize", type=int, default=128, help="Batch size for training")
    parser.add_argument("--timesteps", type=int, default=20000, help="Total training timesteps")
    parser.add_argument("--model-load-path", type=str, default=None, help="Path to load model from")
    parser.add_argument("--model-save-path", type=str, default="ppo_residual_servo", help="Path to save model to")
    parser.add_argument("--gui", action="store_true", help="Enable GUI simulation")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--logdir", type=str, default="runs/residual_rl", help="TensorBoard log directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.train:
        train(
            total_timesteps=args.timesteps,
            n_steps=args.n_steps,
            batchsize=args.batchsize,
            gui=args.gui,
            seed=args.seed,
            model_load_path=args.model_load_path,
            model_save_path=args.model_save_path,
            tensorboard_log=args.logdir,
        )
    if args.eval:
        evaluate(model_load_path=args.model_load_path, episodes=args.episodes, gui=args.gui, seed=args.seed)
