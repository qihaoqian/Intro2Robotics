import casadi
import numpy as np
import utils


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
        self.obstacles = [(-2, -2, 0.5), (1, 2, 0.5)]
        self.robot_radius = 0.3

    def __call__(self, t: int, cur_state: np.ndarray, cur_ref_state: np.ndarray) -> np.ndarray:
        """
        Given the time step, current state, and reference state, return the control input.
        Args:
            t (int): time step
            cur_state (np.ndarray): current state
            cur_ref_state (np.ndarray): reference state
        Returns:
            np.ndarray: control input
        """
        # 初始误差状态
        e0 = np.zeros(3)
        e0[0:2] = cur_state[0:2] - cur_ref_state[0:2]
        e0[2] = (cur_state[2] - cur_ref_state[2] + np.pi) % (2 * np.pi) - np.pi

        opti = casadi.Opti()
        U = opti.variable(2, self.T)  # 控制量序列 [v, w]
        E = opti.variable(3, self.T+1)  # 误差状态序列 [ex, ey, etheta]

        # 初始误差状态约束
        opti.subject_to(E[:,0] == e0)

        cost = 0
        for k in range(self.T):
            # 参考轨迹
            ref = utils.lissajous(t + k)
            ref_next = utils.lissajous(t + k + 1)
            # 误差状态
            error = E[:,k]
            # 控制量
            u = U[:,k]
            # 目标函数
            pos_err = error[0:2]
            theta_err = error[2]
            cost += self.gamma**k * (casadi.mtimes([pos_err.T, self.Q, pos_err]) + self.q * (1 - casadi.cos(theta_err))**2 + casadi.mtimes([u.T, self.R, u]))
            # 动力学递推
            alpha = ref[2]
            delta = self.dt
            theta_mid = error[2] + alpha + 0.5 * u[1] * delta
            eps = 1e-6
            w_delta = 0.5 * u[1] * delta
            sinc = casadi.if_else(casadi.fabs(w_delta) < eps, 1, casadi.sin(w_delta) / w_delta)
            dx = delta * sinc * casadi.cos(theta_mid) * u[0]
            dy = delta * sinc * casadi.sin(theta_mid) * u[0]
            dtheta = u[1] * delta
            d_ref = np.array(ref) - np.array(ref_next)
            d_ref[2] = (d_ref[2] + np.pi) % (2 * np.pi) - np.pi
            e_next = error + casadi.vertcat(dx, dy, dtheta) + casadi.vertcat(d_ref[0], d_ref[1], d_ref[2])
            # 递推约束
            opti.subject_to(E[:,k+1] == e_next)
            # 控制量约束
            opti.subject_to(opti.bounded(self.v_min, u[0], self.v_max))
            opti.subject_to(opti.bounded(self.w_min, u[1], self.w_max))
            # 障碍物约束
            pos = error[0:2] + ref[0:2]
            for obs in self.obstacles:
                obs_center = np.array([obs[0], obs[1]])
                obs_r = obs[2]
                min_dist = obs_r + self.robot_radius
                dist = casadi.sqrt((pos[0] - obs_center[0])**2 + (pos[1] - obs_center[1])**2)
                opti.subject_to(dist >= (min_dist+1e-3))  # 加1e-3防止数值问题
            # 地图边界约束
            opti.subject_to(pos[0] >= -3 + self.robot_radius)
            opti.subject_to(pos[0] <= 3 - self.robot_radius)
            opti.subject_to(pos[1] >= -3 + self.robot_radius)
            opti.subject_to(pos[1] <= 3 - self.robot_radius)

        # 终端代价
        error_T = E[:,self.T]
        cost += self.gamma**self.T * (casadi.mtimes([error_T[0:2].T, self.Q, error_T[0:2]]) + self.q * (1 - casadi.cos(error_T[2]))**2)

        opti.minimize(cost)
        # 设置求解器参数
        p_opts = {
            "print_time": False,
            "expand": True  # 展开问题，可能提高数值稳定性
        }
        s_opts = {
            "print_level": 0,
            "max_iter": 3000,  # 增加最大迭代次数
            "tol": 1e-6,      # 提高精度要求
            "mu_strategy": "adaptive",
            "bound_push": 1e-8,
            "bound_frac": 1e-8,
            "hessian_approximation": "limited-memory",  # 使用L-BFGS近似Hessian
            "linear_solver": "mumps"  # 更换线性求解器
        }
        opti.solver("ipopt", p_opts, s_opts)
        # 初始猜测
        opti.set_initial(U, 0)
        opti.set_initial(E, 0)
        # 求解
        sol = opti.solve()
        u_star = sol.value(U[:,0])
        return u_star
