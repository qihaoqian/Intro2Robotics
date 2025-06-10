from time import time
import numpy as np
import utils
# from mujoco_car import MujocoCarSim
from cec_numpy import CEC


use_mujoco = False


def main():
    # Obstacles in the environment
    obstacles = np.array([[-2, -2, 0.5], [1, 2, 0.5]])

    # Params
    traj = utils.lissajous
    ref_traj = []
    error_trans = 0.0
    error_rot = 0.0
    car_states = []
    times = []

    # 初始化CEC planner
    Q = np.diag([1.0, 1.0])  # 位置误差权重
    R = np.diag([0.1, 0.05])   # 控制输入权重，降低角速度的权重
    cec_planner = CEC(T=5, Q=Q, R=R, q=1, gamma=0.95)  # 增大角度误差权重

    # Start main loop
    main_loop = time()  # return time in sec

    # Initialize state
    cur_state = np.array([utils.x_init, utils.y_init, utils.theta_init])
    cur_iter = 0

    # Initialize Mujoco simulation environment
    mujoco_sim = None
    if use_mujoco:
        mujoco_sim = MujocoCarSim()

    # Main loop
    while cur_iter * utils.time_step < utils.sim_time:
        t1 = time()
        # Get reference state
        cur_time = cur_iter * utils.time_step
        cur_ref = traj(cur_iter)
        # Save current state and reference state for visualization
        ref_traj.append(cur_ref)
        car_states.append(cur_state)

        ################################################################
        # Generate control input
        # TODO: Replace this simple controller with your own controller
        
        # control = utils.simple_controller(cur_state, cur_ref)
        control = cec_planner(cur_iter, cur_state, cur_ref)
        print("[v,w]", control)
        ################################################################

        # Apply control input
        if use_mujoco:
            next_state = mujoco_sim.car_next_state(control)
        else:
            next_state = utils.car_next_state(utils.time_step, cur_state, control, noise=True)

        # Update current state
        cur_state = next_state
        # Loop time
        t2 = utils.time()
        print(cur_iter)
        print(t2 - t1)
        times.append(t2 - t1)
        cur_err = cur_state - cur_ref
        cur_err[2] = np.arctan2(np.sin(cur_err[2]), np.cos(cur_err[2]))
        error_trans = error_trans + np.linalg.norm(cur_err[:2])
        error_rot = error_rot + np.abs(cur_err[2])
        print(cur_err, error_trans, error_rot)
        print("======================")
        cur_iter = cur_iter + 1

    main_loop_time = time()
    print("\n\n")
    print("Total time: ", main_loop_time - main_loop)
    print("Average iteration time: ", np.array(times).mean() * 1000, "ms")
    print("Final error_trains: ", error_trans)
    print("Final error_rot: ", error_rot)

    # Proper shunt down mujoco
    if use_mujoco:
        mujoco_sim.viewer_handle.close()

    # Visualization
    ref_traj = np.array(ref_traj)
    car_states = np.array(car_states)
    times = np.array(times)
    utils.visualize(car_states, ref_traj, obstacles, times, utils.time_step, save=True)


if __name__ == "__main__":
    main()

