import numpy as np
import matplotlib.pyplot as plt
import cv2


resolution = 0.05            # 每个栅格 0.05 米
grid_size = 1000              # 500x500 的地图
cx = grid_size // 2
cy = grid_size // 2
log_odds_map = np.load("log_odds_map_downsample.npy")
trajectory = np.load("lidar_trajectory_downsample.npy")
# log_odds_map = log_odds_map.transpose((1, 0))
prob_map = 1.0 / (1.0 + np.exp(np.clip(-log_odds_map, -10, 10)))
occupancy_grid = np.zeros_like(prob_map, dtype=np.uint8)
occupancy_grid[prob_map > 0.9] = 255  #障碍
occupancy_grid[prob_map < 0.3] = 0    #自由
occupancy_grid[(prob_map >= 0.3) & (prob_map <= 0.9)] = 128  #未知
for (rx, ry, rtheta) in trajectory:
    gx = int(rx / resolution + cx)
    gy = int(ry / resolution + cy)
    if 0 <= gx < grid_size and 0 <= gy < grid_size:
        occupancy_grid[gx, gy] = 255  # 你可以自行选择合适的灰度值

# plt.imshow(occupancy_grid, cmap='gray')
plt.imshow(occupancy_grid, cmap='gray_r')
plt.title("Occupancy Grid (ICP incremental)")
plt.colorbar()
plt.show()
