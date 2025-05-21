import numpy as np
import matplotlib.pyplot as plt

dataset_num = 1

landmarks = np.load(f"landmarks_{dataset_num}.npy")
landmarks_init = np.load(f"landmarks_init_{dataset_num}.npy")
pose = np.load(f"pose_imu_ekf_{dataset_num}.npy")
trajectory_xy = pose[:, 0:2, 3]


fig, ax = plt.subplots()
ax.set_title(f"Landmarks dataset {dataset_num}")

ax.plot(trajectory_xy[:, 0], trajectory_xy[:, 1], label="Estimated Trajectory", color="blue")
ax.scatter(trajectory_xy[0, 0], trajectory_xy[0, 1], marker='s', label="start")
ax.scatter(trajectory_xy[-1, 0], trajectory_xy[-1, 1], marker='o', label="end")
ax.scatter(landmarks[:, 0], landmarks[:, 1], label="Landmarks", color="red", marker="o", s=0.3)
ax.scatter(landmarks_init[:, 0], landmarks_init[:, 1], label="Initial Landmarks", color="green", marker="o", s=0.3)

ax.set_xlabel("X (meters)")
ax.set_ylabel("Y (meters)")
ax.legend()
# ax.set_title("Trajectory and Landmarks (XY)")
ax.axis("equal")  # 保证 x 和 y 轴比例一致

plt.savefig(f"results/landmarks_{dataset_num}.png")
plt.show()
plt.close() 
