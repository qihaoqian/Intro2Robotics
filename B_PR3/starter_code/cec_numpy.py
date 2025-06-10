import numpy as np
import utils
from scipy.optimize import minimize
from typing import List, Tuple

class CEC:
    def __init__(self, T=7, Q=None, R=None, q=1.0, gamma=0.95) -> None:
        self.T = T  # 预测步长
        self.Q = Q if Q is not None else np.eye(2)
        self.R = R if R is not None else np.eye(2)
        self.q = q
        self.gamma = gamma
        self.v_min = utils.v_min
        self.v_max = utils.v_max
        self.w_min = utils.w_min
        self.w_max = utils.w_max
        self.dt = utils.time_step
        # 障碍物参数
        self.obstacles = [(-2, -2, 0.5), (1, 2, 0.5)]
        self.robot_radius = 0.3
        self.last_solution = None

    def dynamics(self, error: np.ndarray, u: np.ndarray, ref: np.ndarray, ref_next: np.ndarray) -> np.ndarray:
        """计算系统动力学"""
        alpha = ref[2]
        delta = self.dt
        theta_mid = error[2] + alpha + 0.5 * u[1] * delta
        w_delta = 0.5 * u[1] * delta
        
        # 计算sinc
        eps = 1e-6
        if abs(w_delta) < eps:
            sinc = 1.0
        else:
            sinc = np.sin(w_delta) / w_delta

        dx = delta * sinc * np.cos(theta_mid) * u[0]
        dy = delta * sinc * np.sin(theta_mid) * u[0]
        dtheta = u[1] * delta

        d_ref = ref - ref_next
        d_ref[2] = (d_ref[2] + np.pi) % (2 * np.pi) - np.pi

        return error + np.array([dx, dy, dtheta]) + d_ref

    def get_trajectory(self, x: np.ndarray, t: int, e0: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """获取完整轨迹，返回状态和位置序列"""
        U = x.reshape(-1, 2)
        error = e0.copy()
        trajectory = []
        
        for k in range(self.T):
            ref = np.array(utils.lissajous(t + k))
            ref_next = np.array(utils.lissajous(t + k + 1))
            u = U[k]
            
            # 保存当前状态和位置
            pos = error[:2] + ref[:2]
            trajectory.append((error.copy(), pos))
            
            # 更新状态
            error = self.dynamics(error, u, ref, ref_next)
        
        # 添加最后一个状态
        pos = error[:2] + np.array(utils.lissajous(t + self.T))[:2]
        trajectory.append((error, pos))
        
        return trajectory

    def cost_function(self, x: np.ndarray, t: int, e0: np.ndarray, cur_ref_state: np.ndarray) -> float:
        """计算代价函数"""
        U = x.reshape(-1, 2)
        cost = 0
        error = e0.copy()

        for k in range(self.T):
            ref = np.array(utils.lissajous(t + k))
            ref_next = np.array(utils.lissajous(t + k + 1))
            u = U[k]
            
            # 计算当前代价
            pos_err = error[:2]
            theta_err = error[2]
            stage_cost = (self.gamma**k * (
                pos_err.T @ self.Q @ pos_err + 
                self.q * (1 - np.cos(theta_err))**2 + 
                u.T @ self.R @ u
            ))
            cost += stage_cost

            # 更新状态
            error = self.dynamics(error, u, ref, ref_next)

        # 终端代价
        cost += self.gamma**self.T * (
            error[:2].T @ self.Q @ error[:2] + 
            self.q * (1 - np.cos(error[2]))**2
        )

        return cost

    def obstacle_constraints(self, x: np.ndarray, t: int, e0: np.ndarray) -> np.ndarray:
        """障碍物约束函数"""
        trajectory = self.get_trajectory(x, t, e0)
        constraints = []
        
        for _, pos in trajectory:
            # 对每个障碍物
            for obs in self.obstacles:
                obs_center = np.array([obs[0], obs[1]])
                obs_r = obs[2]
                min_dist = obs_r + self.robot_radius
                dist = np.sqrt(np.sum((pos - obs_center[:2])**2))
                constraints.append(dist - min_dist)  # >= 0
        
        return np.array(constraints)

    def boundary_constraints(self, x: np.ndarray, t: int, e0: np.ndarray) -> np.ndarray:
        """边界约束函数"""
        trajectory = self.get_trajectory(x, t, e0)
        constraints = []
        
        for _, pos in trajectory:
            # 左边界
            constraints.append(pos[0] - (-3 + self.robot_radius))  # >= 0
            # 右边界
            constraints.append(3 - self.robot_radius - pos[0])     # >= 0
            # 下边界
            constraints.append(pos[1] - (-3 + self.robot_radius))  # >= 0
            # 上边界
            constraints.append(3 - self.robot_radius - pos[1])     # >= 0
        
        return np.array(constraints)

    def get_initial_guess(self, t: int, e0: np.ndarray) -> np.ndarray:
        """生成更好的初始猜测"""
        if self.last_solution is not None:
            # 使用上一次的解，但将控制序列向前移动一步
            x0 = np.zeros(self.T * 2)
            x0[:-2] = self.last_solution[2:]  # 移除第一个控制输入，后面补零
            return x0
        
        # 如果没有上一次的解，生成一个简单的初始猜测
        x0 = np.zeros(self.T * 2)
        for i in range(0, self.T * 2, 2):
            x0[i] = (self.v_max + self.v_min) / 2  # v设置为速度范围的中间值
            x0[i+1] = 0  # w初始设为0
        return x0

    def __call__(self, t: int, cur_state: np.ndarray, cur_ref_state: np.ndarray) -> np.ndarray:
        """给定时间步、当前状态和参考状态，返回控制输入"""
        # 计算初始误差状态
        e0 = np.zeros(3)
        e0[0:2] = cur_state[0:2] - cur_ref_state[0:2]
        e0[2] = (cur_state[2] - cur_ref_state[2] + np.pi) % (2 * np.pi) - np.pi

        # 设置优化问题的边界（控制输入约束）
        bounds = []
        for _ in range(self.T):
            bounds.extend([
                (self.v_min, self.v_max),  # v的边界
                (self.w_min, self.w_max)   # w的边界
            ])

        # 获取初始猜测
        x0 = self.get_initial_guess(t, e0)

        # 定义非线性约束
        obstacle_constraint = {
            'type': 'ineq',
            'fun': lambda x: self.obstacle_constraints(x, t, e0)
        }
        
        boundary_constraint = {
            'type': 'ineq',
            'fun': lambda x: self.boundary_constraints(x, t, e0)
        }

        # 使用SLSQP求解器进行优化
        result = minimize(
            self.cost_function,
            x0,
            args=(t, e0, cur_ref_state),
            method='SLSQP',
            bounds=bounds,
            constraints=[obstacle_constraint, boundary_constraint],
            options={
                'maxiter': 1000,     # 显著增加最大迭代次数
                'ftol': 1e-6,        # 函数收敛容差
                'eps': 1e-8,         # 数值微分步长
                'disp': False,
                'iprint': 1,         # 打印优化过程信息
                'finite_diff_rel_step': 1e-8  # 有限差分相对步长
            }
        )

        if not result.success:
            print(f"警告：优化未成功收敛！状态: {result.message}")
            print(f"迭代次数: {result.nit}")
            print(f"约束违反程度: {result.maxcv if hasattr(result, 'maxcv') else 'N/A'}")

        # 保存当前解作为下一次的初始猜测
        self.last_solution = result.x

        # 返回最优控制序列的第一个控制输入
        return result.x[:2] 