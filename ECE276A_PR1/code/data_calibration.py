import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import quat2euler, mat2euler
from scipy.optimize import minimize
import jax.numpy as jnp
from jax import grad, jit, vmap
import jax
from jax_calculate import qexp, qmult, qinverse, safe_log_quaternion, log_quaternion


import os
import sys
import numpy as np
from transforms3d.quaternions import mat2quat, quat2mat
from transforms3d.axangles import mat2axangle, axangle2mat
from read_data import load_imu_dataset, load_vicon_dataset, load_cam_dataset


def compute_bias(sensor_data, static_range=100):
    """
    Compute bias from static period (default first static_range frames)
    sensor_data: shape = (3, N)
    """
    bias = np.mean(sensor_data[:, :static_range], axis=1)  
    return bias


def so3_log(R):
    """
    Compute the logarithmic map of SO(3) matrix R, returning axis*theta
    suitable for numerical differentiation of rotation matrices.
    """
    tr = np.trace(R)
    theta = np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))  # ensure input is in [-1,1]

    if abs(theta) < 1e-6:
        return np.zeros(3)  # approximation as zero vector

    if abs(theta - np.pi) < 1e-6:
        # handle nearly 180-degree rotations, avoiding singularities
        return np.array([
            np.sign(R[2,1] - R[1,2]),
            np.sign(R[0,2] - R[2,0]),
            np.sign(R[1,0] - R[0,1])
        ]) * theta

    w_hat = (R - R.T) / (2 * np.sin(theta))  # compute rotation axis
    return np.array([w_hat[2,1], w_hat[0,2], w_hat[1,0]]) * theta

def numeric_angular_velocity(Rs, ts):
    """
    Compute angular velocity from a sequence of rotation matrices Rs[:,:,i] at timestamps ts[:,i]
    returns w_body: shape=(3, N-1), the average angular velocity (IMU coordinate system)
    Note: if VICON and IMU coordinate systems are different, do extra transformations.
    """
    N = Rs.shape[2]
    w_body = []
    for i in range(N-1):
        Rk = Rs[:,:,i]
        Rk1 = Rs[:,:,i+1]
        dR = Rk.T @ Rk1  # relative rotation in Rk coordinates
        angle = np.arccos((np.trace(dR) - 1) / 2)
        if angle > np.pi / 2:
            print(f"Warning: Large rotation between frames at index {i}, angle={angle} rad")
        dt = ts[:, i+1] - ts[:, i]
        if dt < 1e-3:  # e.g. if dt is nearly zero
            print(f"Warning: dt too small at index {i}, dt={dt}")
        w_local = so3_log(dR) / dt  # axis*theta / dt
        if np.linalg.norm(w_local) > 10:  # set a reasonable angular velocity threshold
            print(f"Warning: Unusually high angular velocity at frame {i}: {w_local}")
        # This is the rotation vector in Rk coordinates, since dR = exp([w_local^] * dt).
        # If Rk is the IMU coordinate system, w_local is the IMU angular velocity.
        # If transformations are needed, do them here, e.g. R_align.
        w_body.append(w_local)
    return np.array(w_body).T  # shape=(3, N-1)


def compute_accel_scale_factors(accel_meas, R_vicon, g=9.81):
    """
    Compute acceleration scale factors using the "ignore centripetal acceleration" method
    Parameters:
    - accel_meas: shape = (3, N), already de-biased IMU acceleration measurements (unit can be roughly converted to m/s^2)
    - R_vicon: shape = (3,3,N), rotation matrices from IMU to world coordinates (or vice versa, ensure consistency)
    - g: gravity acceleration scalar, default is 9.81

    Returns:
    - scale_factors: shape=(3,), acceleration scale factors for x, y, z axes
    """
    N = accel_meas.shape[1]
    # here we assume R_vicon[:,:,i] converts IMU coordinates to world coordinates
    # so the "true" acceleration in IMU coordinates is R^T * [0,0,g]
    a_true_all = []
    a_meas_all = []

    for i in range(N):
        R = R_vicon[:, :, i]
        a_true_imu = R.T @ np.array([0, 0, g])  # IMU coordinates
        a_true_all.append(a_true_imu)
        a_meas_all.append(accel_meas[:, i])

    a_true_all = np.array(a_true_all).T   # shape=(3, N)
    a_meas_all = np.array(a_meas_all).T   # shape=(3, N)

    scale_factors = np.zeros(3)
    for axis in range(3):
        num = np.sum(a_true_all[axis,:] * a_meas_all[axis,:])
        den = np.sum(a_meas_all[axis,:] ** 2)
        if abs(den) < 1e-8:
            scale_factors[axis] = 1.0
        else:
            scale_factors[axis] = num / den
    
    return scale_factors

def test_accel_data_calibration(timestamps, gyro_data, accel_data, dt, vicd):
    N = len(timestamps)
    static_range = 10
    R_vicon = vicd['rots']
    roted_accel = np.einsum('ijk, jk->ik', R_vicon, accel_data)
    fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(timestamps, roted_accel[0])
    ax[1].plot(timestamps, roted_accel[1])
    ax[2].plot(timestamps, roted_accel[2])
    ax[0].set_title("X axis")
    ax[1].set_title("Y axis")
    ax[2].set_title("Z axis")
    plt.show()

def test_gyro_data_calibration(timestamps, gyro_data, vicon_ts, R_vicon, scale_factor_gyro):
    N = len(timestamps)
    w_vicon = numeric_angular_velocity(R_vicon, vicon_ts)   # shape=(3, N-1)
    gyro_data = gyro_data

    fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True) 
    ax[0].plot(range(len(timestamps)-1), w_vicon[0], label="vicon")
    ax[0].plot(range(len(timestamps)), gyro_data[0], label="imu")
    ax[1].plot(range(len(timestamps)-1), w_vicon[1], label="vicon")
    ax[1].plot(range(len(timestamps)), gyro_data[1], label="imu")
    ax[2].plot(range(len(timestamps)-1), w_vicon[2], label="vicon")
    ax[2].plot(range(len(timestamps)), gyro_data[2], label="imu")
    ax[0].set_title("X axis")
    ax[1].set_title("Y axis")
    ax[2].set_title("Z axis")
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.show()

def import_data(dataset_number):

    imud = load_imu_dataset(dataset_number)
    vicd = load_vicon_dataset(dataset_number)
    # ------------------------------
    # 2) Read VICON data
    # ------------------------------
    vicon_ts = vicd['ts']   # Assuming vicon_ts is UNIX timestamp (1,N)
    R_vicon = vicd['rots']  # Shape=(3,3,N)

    # ------------------------------
    # 1) Read IMU data
    # ------------------------------
    length = min(vicon_ts.shape[1], imud.shape[1])  # Align data length based on VICON timestamps
    imud = imud[:, :length]
    vicon_ts = vicon_ts[:, :length]
    R_vicon = R_vicon[:, :, :length]  # Align data length

    # ------------------------------
    # 3) Remove data points with small dt
    # ------------------------------
    dt_vicon = np.diff(vicon_ts, prepend= np.inf)  # Calculate time intervals (seconds)

    # **If vicon_ts is stored in milliseconds, convert to seconds**
    # vicon_ts = vicon_ts / 1000  # If vicon_ts is ms, convert to s

    # Find data points with small dt
    mask = dt_vicon >= 1e-4  # True means keep, False means delete
    mask = mask.flatten()

    # Delete corresponding data points
    vicon_ts = vicon_ts[:, mask]
    R_vicon = R_vicon[:, :, mask]  # Delete corresponding rotation matrices
    imud = imud[:, mask]           # Delete corresponding IMU data

    # ------------------------------
    # 4) Extract timestamps, gyro and accel data
    # ------------------------------
    timestamps = imud[0]           # IMU timestamps
    gyro_data_raw = imud[4:7, :]   # Gyro raw ADC data
    accel_data_raw = imud[1:4, :]  # Accel raw ADC data

    # ------------------------------
    # 3) Calibrate data using bias and scale
    # ------------------------------
    # Static segment length
    static_range = 500

    # 3.1) Gyro bias
    bias_gyro_adc = compute_bias(gyro_data_raw, static_range=static_range)

    # 3.3) Some constants (need to match sensor model)
    VREF = 3000.0            # ADC reference voltage (mV)
    ADC_MAX = 1023.0         # 10-bit
    G = 9.81                 # m/s^2
    # Assume ADXL335 sensitivity is 300 mV/g; gyro (e.g. L3G4200D) nominal value
    ACCEL_SENS_MV_G = 300.0  
    GYRO_SENS_MV_DPS = 3.33  # mV/(deg/s), later convert to rad/s

    # 3.4) Remove bias and convert to physical units
    scale_factor_gyro = VREF/ADC_MAX/GYRO_SENS_MV_DPS * (np.pi/180.0)  # rad/s
    gyro_data_calibrated = 4 * (gyro_data_raw - bias_gyro_adc[:, None]) * scale_factor_gyro  # rad/s

    # 3.2) Accel bias
    scale_factor_accel = VREF/ADC_MAX/ACCEL_SENS_MV_G
    bias_accel_adc = np.array([-512.02051, -500.68578365, 502.8718279])
    
    accel_data_calibrated = (accel_data_raw - bias_accel_adc[:, None]) * scale_factor_accel  # Convert to g units

    # ------------------------------
    # 7) Return processed data
    # ------------------------------
    dt_imu = np.diff(timestamps)  # IMU time differences

    return (timestamps, 
            gyro_data_calibrated, 
            accel_data_calibrated, 
            dt_imu, 
            R_vicon,
            vicon_ts
            )

def import_test_data(dataset_number):
    imud = load_imu_dataset(dataset_number)

    # ------------------------------
    # 4) Extract timestamps, gyro and accel data
    # ------------------------------
    timestamps = imud[0]           # IMU timestamps
    gyro_data_raw = imud[4:7, :]   # Gyro raw ADC data
    accel_data_raw = imud[1:4, :]  # Accel raw ADC data

    # ------------------------------
    # 3) Calibrate data using bias and scale
    # ------------------------------
    # Static segment length
    static_range = 500

    # 3.1) Gyro bias
    bias_gyro_adc = compute_bias(gyro_data_raw, static_range=static_range)

    # 3.3) Some constants (need to match sensor model)
    VREF = 3000.0            # ADC reference voltage (mV)
    ADC_MAX = 1023.0         # 10-bit
    G = 9.81                 # m/s^2
    # Assume ADXL335 sensitivity is 300 mV/g; gyro (e.g. L3G4200D) nominal value
    ACCEL_SENS_MV_G = 300.0  
    GYRO_SENS_MV_DPS = 3.33  # mV/(deg/s), later convert to rad/s

    # 3.4) Remove bias and convert to physical units
    scale_factor_gyro = VREF/ADC_MAX/GYRO_SENS_MV_DPS * (np.pi/180.0)  # rad/s
    gyro_data_calibrated = 4 * (gyro_data_raw - bias_gyro_adc[:, None]) * scale_factor_gyro  # rad/s

    # 3.2) Accel bias
    scale_factor_accel = VREF/ADC_MAX/ACCEL_SENS_MV_G
    bias_accel_adc = np.array([-512.02051, -500.68578365, 502.8718279])
    
    accel_data_calibrated = (accel_data_raw - bias_accel_adc[:, None]) * scale_factor_accel  # Convert to g units

    # ------------------------------
    # 7) Return processed data
    # ------------------------------
    dt_imu = np.diff(timestamps)  # IMU time differences

    return (timestamps, 
            gyro_data_calibrated, 
            accel_data_calibrated, 
            dt_imu, 
            )
# timestamps, gyro_data, accel_data, dt_imu, R_vicon, vicon_ts = import_data("1")

# test_accel_data_calibration(timestamps, gyro_data, accel_data, dt, vicd)
