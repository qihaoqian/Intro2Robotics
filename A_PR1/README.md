# ECE 276A Project 1



## Overview

This project involves quaternion-based motion estimation using IMU data and constructing panoramas from camera images with VICON data. It utilizes JAX for gradient-based optimization of quaternion trajectories and applies transformation matrices to align sensor data.

## Repository Structure

- **`problem1.py`**: Implements quaternion-based motion estimation using JAX, including functions for quaternion optimization, error computation, and visualization of estimated orientations.
- **`problem2.py`**: Implements panorama stitching using camera images and VICON data, leveraging cylindrical and spherical projections.
- **`read_data.py`**: Functions to read IMU, VICON, and camera datasets from pickle files.
- **`data_calibration.py`**: Calibration functions for IMU data, including gyro and accelerometer bias estimation and scale factor adjustments.
- **`jax_calculate.py`**: Utility functions for quaternion operations, including multiplication, exponential/logarithmic maps, and conversions between quaternions and Euler angles/matrices.

## Dependencies

Ensure the following Python libraries are installed:

```
pip install numpy matplotlib jax jaxlib scipy opencv-python tqdm transforms3d pickle5
```

## Usage

### 1. Quaternion-Based Motion Estimation

To estimate motion from IMU data:

```sh
python problem1.py
```

This script:

- Reads IMU and VICON datasets.
- Optimizes quaternion trajectories based on IMU readings.
- Converts quaternions to Euler angles.
- Plots predictions against ground truth data.

### 2. Panorama Construction

To build a panorama using camera images and VICON transformations:

```sh
python problem2.py
```

This script:

- Reads camera images and VICON datasets.
- Computes transformations to align images.
- Stitches images into a cylindrical panorama.
- Saves the resulting panorama as an image.

## Data Structure

### IMU Data

- **Timestamps** (`timestamps`)
- **Gyroscope readings** (`gyro_data`)
- **Accelerometer readings** (`accel_data`)
- **Time intervals** (`dt_imu`)

### VICON Data

- **Rotation matrices** (`R_vicon`)
- **Timestamps** (`vicon_ts`)

### Camera Data

- **Images** (`cam_data`)
- **Timestamps** (`cam_ts`)

## Output

- **Estimated quaternion trajectory** (`quaternion_estimates`)
- **Euler angle plots** (`euler_angles_pred_<dataset>.png`)
- **Panorama images** (`panorama_<dataset>.png`)

## Future Improvements

- Enhance quaternion optimization with adaptive learning rates.
- Improve panorama blending for seamless image transitions.
- Implement real-time quaternion estimation and visualization.

## License

This project is licensed under the MIT License.

