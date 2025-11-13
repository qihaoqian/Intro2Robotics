import sys
import os
import numpy as np
import matplotlib.pyplot as plt
# from transforms3d.euler import quat2euler, mat2euler
from scipy.optimize import minimize
import jax.numpy as jnp
from jax import grad, jit, vmap
import jax
from jax_calculate import qexp, qmult, qinverse, safe_log_quaternion, quat2euler, mat2euler, quat2mat
from data_calibration import import_data, import_test_data

jax.config.update("jax_enable_x64", True)


def predict_next_quaternion(q_t, dt, gyro_data):
    q_exp_arg = jnp.concatenate([jnp.array([0]), dt * gyro_data / 2])
    exp_q = qexp(q_exp_arg)
    return qmult(q_t, exp_q)

def calculate_acceleration(q_t):
    g = 9.81
    q_gravity = jnp.array([0, 0, 0, 1])
    q_inverse = qinverse(q_t)
    q1 = qmult(q_inverse, q_gravity)
    calculated_a = qmult(q1, q_t)
    return calculated_a[1:]

def batch_cost1(q_t, q_t_next, dt, gyro_data):
    """Calculate quaternion error term"""
    predicted_q = predict_next_quaternion(q_t, dt, gyro_data)
    # Compute the logarithmic map of the prediction error
    error = safe_log_quaternion(qmult(qinverse(q_t_next), predicted_q))
    if error is None:
        return 1e-10
    # Return the squared norm of the error
    return jnp.linalg.norm(2 * error) ** 2

def batch_cost2(q_t, a_t):
    """Calculate acceleration error term"""
    calculated_a = calculate_acceleration(q_t)
    # To make the acceleration into quaternion form, concatenate 0 as the scalar part
    a_quat = jnp.concatenate([jnp.array([0.0]), a_t])
    calculated_a_quat = jnp.concatenate([jnp.array([0.0]), calculated_a])
    diff = a_quat - calculated_a_quat
    if diff is None:
        return 1e-10
    # Compute the squared norm of the error
    return jnp.linalg.norm(diff) ** 2

def cost_function(q_list_flat, dt, gyro_data, accel_data):
    """
    q_list_flat: A 1D array of shape (T*4,), will be reshaped internally to (T,4)
    timestamps: A 1D array of length T
    gyro_data: Original gyroscope data, shape (3, T), needs to be transposed to (T,3)
    accel_data: Original acceleration data, shape (3, T), needs to be transposed to (T,3)
    """
    T = len(dt) + 1
    q_list = q_list_flat.reshape((T, 4))

    # Transpose data so each row corresponds to a time step
    gyro_data = gyro_data.T  # Now shape (T, 3)
    accel_data = accel_data.T  # Now shape (T, 3)

    # Use vmap to compute quaternion error cost1 for each time step
    # Corresponding parameter shapes are:
    #   q_t: (T-1, 4)
    #   q_t_next: (T-1, 4)
    #   tau: (T-1,) - scalar
    #   omega_t: (T-1, 3)
    cost1 = jnp.sum(
        vmap(batch_cost1)(
            q_list[:-1],
            q_list[1:],
            dt[0:],
            gyro_data[:-1]
        )
    )

    # Use vmap to compute acceleration error cost2 for each time step
    # Here q_list[1:] corresponds to the same time step as accel_data[1:]
    cost2 = jnp.sum(
        vmap(batch_cost2)(
            q_list[1:],
            accel_data[1:]
        )
    )

    return 0.5 * cost1 + 0.5 * cost2

@jit  # JIT compilation optimization
def cost_function_jit(q_list_flat, timestamps, gyro_data, accel_data):
    return cost_function(q_list_flat, timestamps, gyro_data, accel_data)

def optimize_quaternion(q_list_jax, dt, gyro_data, accel_data):
    learning_rate = 1e-3
    num_iterations = 2500

    grad_fn = grad(cost_function_jit)

    for i in range(num_iterations):
        grad_values = grad_fn(q_list_jax, dt, gyro_data, accel_data)
        q_list_jax -= learning_rate * grad_values

        # Normalize quaternions
        q_list_jax = q_list_jax.reshape((-1, 4))
        norms = jnp.linalg.norm(q_list_jax, axis=1, keepdims=True)
        q_list_jax = q_list_jax / (norms + 1e-8)
        q_list_jax = q_list_jax.flatten()

        if i % 100 == 0:
            loss_value = cost_function_jit(q_list_jax, dt, gyro_data, accel_data)
            print(f"Iteration {i}, Loss: {loss_value}")

    optimized_q = q_list_jax.reshape((len(dt) + 1, 4))
    return np.array(optimized_q)  # Convert JAX array to NumPy array and return

def convert_quat2euler(quaternion_estimates, R_vicon):
    # 将四元数转换为欧拉角
    euler_angles_pred = np.zeros((len(quaternion_estimates), 3))
    for i in range(len(quaternion_estimates)):
        euler_angles_pred[i] = quat2euler(quaternion_estimates[i])

    # Extract Roll, Pitch, Yaw
    roll_pred = euler_angles_pred[:, 0]   # Roll (X-axis rotation)
    pitch_pred = euler_angles_pred[:, 1]  # Pitch (Y-axis rotation)
    yaw_pred = euler_angles_pred[:, 2]    # Yaw (Z-axis rotation)

    # 从 VICON 数据提取地面真实值
    vicon_euler = mat2euler(R_vicon)

    roll_gt = vicon_euler[:, 0]   # Roll (X-axis rotation)
    pitch_gt = vicon_euler[:, 1]  # Pitch (Y-axis rotation)
    yaw_gt = vicon_euler[:, 2]    # Yaw (Z-axis rotation)

    return roll_pred, pitch_pred, yaw_pred, roll_gt, pitch_gt, yaw_gt

def plot_euler_angles(roll_pred, pitch_pred, yaw_pred, roll_gt, pitch_gt, yaw_gt, timestamps, vicon_ts , dataset_num):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    vicon_ts = vicon_ts.reshape(-1,)
    start = min(timestamps[0], vicon_ts[0])
    timestamps_plot = timestamps - start
    vicon_ts_plot = vicon_ts - start

    axes[0].plot(timestamps_plot, roll_pred, label='Prediction')
    axes[0].plot(vicon_ts_plot, roll_gt, label='Ground Truth')
    axes[0].set_title('Roll')
    axes[0].legend()

    axes[1].plot(timestamps_plot, pitch_pred, label='Prediction')
    axes[1].plot(vicon_ts_plot, pitch_gt, label='Ground Truth')
    axes[1].set_title('Pitch')
    axes[1].legend()

    axes[2].plot(timestamps_plot, yaw_pred, label='Prediction')
    axes[2].plot(vicon_ts_plot, yaw_gt, label='Ground Truth')
    axes[2].set_title('Yaw')
    axes[2].legend()

    fig.suptitle(f"Dataset {dataset_num} Prediction vs Ground Truth")
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"euler_angles_pred_{dataset_num}.png")

def predict_testset(dataset_num):
    timestamps, gyro_data, accel_data, dt_imu = import_test_data(dataset_num)
    q_t = jnp.array([1, 0, 0, 0], dtype=jnp.float64)  
    q_list = jnp.tile(q_t, (len(timestamps), 1)).flatten()

    gyro_data = jnp.array(gyro_data, dtype=jnp.float64)
    accel_data = jnp.array(accel_data, dtype=jnp.float64)
    dt_imu = jnp.array(dt_imu, dtype=jnp.float64)  

    quaternion_estimates = optimize_quaternion(q_list, dt_imu, gyro_data, accel_data)
    R_estimates = np.zeros((3, 3, len(quaternion_estimates)))
    for i in range(len(quaternion_estimates)):
        R_estimates[:, :, i] = quat2mat(quaternion_estimates[i])
    vicon_data = {'rots': R_estimates, 'ts': timestamps[np.newaxis, :]}
    import pickle

    with open(f"data/vicon/viconRot{dataset_num}.p", 'wb') as f:
        pickle.dump(vicon_data, f)

    # 将四元数转换为欧拉角
    euler_angles_pred = np.zeros((len(quaternion_estimates), 3))
    for i in range(len(quaternion_estimates)):
        euler_angles_pred[i] = quat2euler(quaternion_estimates[i])

    # Extract Roll, Pitch, Yaw
    roll_pred = euler_angles_pred[:, 0]   # Roll (X-axis rotation)
    pitch_pred = euler_angles_pred[:, 1]  # Pitch (Y-axis rotation)
    yaw_pred = euler_angles_pred[:, 2]    # Yaw (Z-axis rotation)

    plot_testset(roll_pred, pitch_pred, yaw_pred, timestamps, dataset_num)

    return quaternion_estimates

def plot_testset(roll_pred, pitch_pred, yaw_pred, timestamps, dataset_num):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    timestamps_plot = timestamps - timestamps[0]
    
    axes[0].plot(timestamps_plot, roll_pred, label='Prediction')
    axes[0].set_title('Roll')
    axes[0].legend()

    axes[1].plot(timestamps_plot, pitch_pred, label='Prediction')
    axes[1].set_title('Pitch')
    axes[1].legend()

    axes[2].plot(timestamps_plot, yaw_pred, label='Prediction')
    axes[2].set_title('Yaw')
    axes[2].legend()
    fig.suptitle(f"Dataset {dataset_num} Prediction")
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"euler_angles_pred_{dataset_num}.png")

def predict(dataset_num):
    timestamps, gyro_data, accel_data, dt_imu, R_vicon, vicon_ts = import_data(dataset_num)
    q_t = jnp.array([1, 0, 0, 0], dtype=jnp.float64)  
    q_list = jnp.tile(q_t, (len(timestamps), 1)).flatten()

    gyro_data = jnp.array(gyro_data, dtype=jnp.float64)
    accel_data = jnp.array(accel_data, dtype=jnp.float64)
    dt_imu = jnp.array(dt_imu, dtype=jnp.float64)  

    quaternion_estimates = optimize_quaternion(q_list, dt_imu, gyro_data, accel_data)

    roll_pred, pitch_pred, yaw_pred, roll_gt, pitch_gt, yaw_gt = convert_quat2euler(quaternion_estimates, R_vicon)

    plot_euler_angles(roll_pred, pitch_pred, yaw_pred, roll_gt, pitch_gt, yaw_gt, timestamps, vicon_ts, dataset_num)

if __name__ == "__main__":
    # for i in range(1,10):
    #     predict(i)
    # for i in [10, 11]:
    #     predict_testset(i)
    predict(9)








