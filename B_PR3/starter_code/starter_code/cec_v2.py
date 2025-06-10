import casadi as ca
import numpy as np
import utils


class CEC:
    """CasADi‑based compact NMPC controller with soft obstacle / boundary constraints.

    The original implementation could exit with an `Infeasible_Problem_Detected` status
    whenever the hard distance or map‑boundary constraints could not be satisfied for
    the entire horizon.  We (1) turn those constraints into *soft* ones by introducing
    non‑negative slack variables that are quadratically penalised in the cost, (2) allow
    the longitudinal speed to drop all the way to 0 m/s (previous lower‑bound was 0.1 m/s
    and frequently rendered the problem infeasible), and (3) replace the costly `sqrt`
    in the obstacle distance calculation with its squared counterpart to keep the NLP
    smoother.  In addition we patch minor type issues (`list` → CasADi `DM`) and supply a
    graceful fall‑back to the simple P controller when IPOPT still fails.
    """

    def __init__(
        self,
        T: int = 7,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
        q: float = 1.0,
        gamma: float = 0.95,
    ) -> None:
        self.T = T  # prediction horizon
        self.Q = Q if Q is not None else np.eye(2)
        self.R = R if R is not None else np.eye(2)
        self.q = q
        self.gamma = gamma

        # input bounds (zero speed now allowed)
        self.v_min = 0.0
        self.v_max = utils.v_max
        self.w_min = utils.w_min
        self.w_max = utils.w_max

        self.dt = utils.time_step

        # static world description
        self.obstacles = [(-2.0, -2.0, 0.5), (1.0, 2.0, 0.5)]  # (cx, cy, radius)
        self.robot_radius = 0.30

    # ---------------------------------------------------------------------
    # callable interface ---------------------------------------------------
    # ---------------------------------------------------------------------
    def __call__(
        self, t: int, cur_state: np.ndarray, cur_ref_state: np.ndarray
    ) -> np.ndarray:
        """Compute the control command [v, w] at discrete time *t*.

        Parameters
        ----------
        t : int
            Current iteration index (0‑based).
        cur_state : np.ndarray, shape=(3,)
            Current vehicle pose *(x, y, θ)* in the world frame.
        cur_ref_state : np.ndarray, shape=(3,)
            Current reference pose.
        """

        # ------------------------------------------------------------------
        # set‑up
        # ------------------------------------------------------------------
        # initial tracking error (world frame)
        e0 = np.zeros(3)
        e0[:2] = cur_state[:2] - cur_ref_state[:2]
        e0[2] = (cur_state[2] - cur_ref_state[2] + np.pi) % (2 * np.pi) - np.pi

        opti = ca.Opti()
        U = opti.variable(2, self.T)     # optimisation variables: control sequence
        E = opti.variable(3, self.T + 1)  # optimisation variables: error trajectory

        opti.subject_to(E[:, 0] == e0)    # fix initial error

        # ------------------------------------------------------------------
        # stage / terminal costs and constraints
        # ------------------------------------------------------------------
        cost = 0.0
        slack_penalty = 1e4  # weight for *all* slack variables
        slacks: list[ca.MX] = []

        for k in range(self.T):
            # reference poses (as CasADi constants)
            ref_k = utils.lissajous(t + k)
            ref_kp1 = utils.lissajous(t + k + 1)
            ref = ca.DM(ref_k)
            ref_next = ca.DM(ref_kp1)

            err_k = E[:, k]
            u_k = U[:, k]

            # ------------------------- running cost ----------------------
            pos_err = err_k[:2]
            theta_err = err_k[2]
            cost += (
                self.gamma**k
                * (
                    ca.mtimes([pos_err.T, self.Q, pos_err])
                    + self.q * (1 - ca.cos(theta_err)) ** 2
                    + ca.mtimes([u_k.T, self.R, u_k])
                )
            )

            # ------------------------- dynamics --------------------------
            dt = self.dt
            theta_mid = err_k[2] + ref[2] + 0.5 * u_k[1] * dt
            w_dt_half = 0.5 * u_k[1] * dt
            sinc = ca.if_else(ca.fabs(w_dt_half) < 1e-7, 1.0, ca.sin(w_dt_half) / w_dt_half)

            dx = dt * sinc * ca.cos(theta_mid) * u_k[0]
            dy = dt * sinc * ca.sin(theta_mid) * u_k[0]
            dtheta = u_k[1] * dt

            d_ref = ref - ref_next
            d_ref[2] = ca.fmod(d_ref[2] + ca.pi, 2*ca.pi) - ca.pi

            opti.subject_to(E[:, k + 1] == err_k + ca.vertcat(dx, dy, dtheta) + d_ref)

            # ------------------------- input bounds ----------------------
            opti.subject_to(opti.bounded(self.v_min, u_k[0], self.v_max))
            opti.subject_to(opti.bounded(self.w_min, u_k[1], self.w_max))

            # --------------------- obstacle avoidance --------------------
            pos_world = err_k[:2] + ref[:2]
            for cx, cy, r in self.obstacles:
                min_dist = r + self.robot_radius + 1e-3
                dist_sq = (pos_world[0] - cx) ** 2 + (pos_world[1] - cy) ** 2

                slack = opti.variable()
                opti.subject_to(slack >= 0)
                opti.subject_to(dist_sq + slack >= min_dist**2)
                slacks.append(slack)

            # --------------------- map boundary (soft) -------------------
            xmin, xmax = -3 + self.robot_radius, 3 - self.robot_radius
            ymin, ymax = -3 + self.robot_radius, 3 - self.robot_radius

            sx_l = opti.variable(); opti.subject_to(sx_l >= 0)
            sx_r = opti.variable(); opti.subject_to(sx_r >= 0)
            sy_b = opti.variable(); opti.subject_to(sy_b >= 0)
            sy_t = opti.variable(); opti.subject_to(sy_t >= 0)

            opti.subject_to(pos_world[0] + sx_l >= xmin)
            opti.subject_to(pos_world[0] - sx_r <= xmax)
            opti.subject_to(pos_world[1] + sy_b >= ymin)
            opti.subject_to(pos_world[1] - sy_t <= ymax)

            slacks += [sx_l, sx_r, sy_b, sy_t]

        # terminal cost ----------------------------------------------------
        err_T = E[:, self.T]
        cost += self.gamma**self.T * (
            ca.mtimes([err_T[:2].T, self.Q, err_T[:2]])
            + self.q * (1 - ca.cos(err_T[2])) ** 2
        )

        # penalise slacks (quadratic)
        if slacks:
            cost += slack_penalty * ca.sumsqr(ca.vcat(slacks))

        opti.minimize(cost)

        # ------------------------------------------------------------------
        # solver configuration
        # ------------------------------------------------------------------
        opti.solver(
            "ipopt",
            {"expand": True, "print_time": False},
            {
                "print_level": 0,
                "max_iter": 1000,
                "mu_strategy": "adaptive",
                "hessian_approximation": "limited‑memory",
            },
        )

        # initial guess -----------------------------------------------------
        opti.set_initial(U, 0)
        opti.set_initial(E, 0)

        # ------------------------------------------------------------------
        # solve / fall‑back
        # ------------------------------------------------------------------
        try:
            sol = opti.solve()
            u_star = sol.value(U[:, 0])
        except RuntimeError:  # IPOPT failed – use simple P controller
            u_star = utils.simple_controller(cur_state, cur_ref_state)

        return np.array(u_star).flatten()
