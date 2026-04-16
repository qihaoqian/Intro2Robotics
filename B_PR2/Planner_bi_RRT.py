import numpy as np
from collision_test import is_segment_collision_free


class MyPlanner:
    __slots__ = [
        "boundary",
        "blocks",
        "res",
        "step",
        "max_iter",
        "p_goal",
    ]

    def __init__(
        self,
        boundary,
        blocks,
        res=0.5,
        step=None,
        max_iter=500_000,
        p_goal=0.05,
    ):
        self.boundary = np.asarray(boundary, dtype=float).reshape(-1)
        self.blocks = np.asarray(blocks, dtype=float)
        self.res = float(res)
        self.step = float(step) if step is not None else float(res)
        self.max_iter = int(max_iter)
        self.p_goal = float(p_goal)

    def _inside(self, p: np.ndarray) -> bool:
        return bool(np.all(p >= self.boundary[:3]) and np.all(p <= self.boundary[3:6]))

    def _sample(self, root: np.ndarray) -> np.ndarray:
        """Randomly sample a point inside the boundary."""
        if np.random.rand() < self.p_goal:
            # With small prob, sample the *root* of the opposite tree
            return root.copy()
        return np.random.uniform(self.boundary[:3], self.boundary[3:6])

    @staticmethod
    def _nearest(V: np.ndarray, n: int, q_rand: np.ndarray) -> int:
        """Return index of the vertex in V[:n] closest to q_rand.

        V is a pre-allocated (cap, 3) array; only the first n rows are valid.
        Avoids list→ndarray conversion on every call.
        """
        dists = np.linalg.norm(V[:n] - q_rand, axis=1)
        return int(dists.argmin())

    def _steer(self, q_near: np.ndarray, q_rand: np.ndarray) -> np.ndarray | None:
        """Return a new configuration stepping from q_near toward q_rand."""
        v = q_rand - q_near
        d = np.linalg.norm(v)
        if d < 1e-12:
            return None
        return q_near + v * (min(self.step, d) / d)

    def _collision_free(self, p: np.ndarray, q: np.ndarray) -> bool:
        return is_segment_collision_free(p, q, self.blocks, self.boundary)

    @staticmethod
    def _backtrack(V: np.ndarray, P: np.ndarray, i: int) -> list:
        """Collect vertices from index i back to the root (P[root] == -1)."""
        pts = []
        while i != -1:
            pts.append(V[i].copy())
            i = P[i]
        return pts

    def plan(self, start, goal):
        start = np.asarray(start, dtype=float)
        goal  = np.asarray(goal,  dtype=float)

        # Pre-allocate vertex/parent arrays — avoids repeated list→ndarray copies
        # in _nearest. cap is a safe upper bound; in practice trees stay far smaller.
        cap = self.max_iter + 2

        Va = np.empty((cap, 3), dtype=float); Va[0] = start; na = 1
        Pa = np.full(cap, -1, dtype=np.int32)
        Vb = np.empty((cap, 3), dtype=float); Vb[0] = goal;  nb = 1
        Pb = np.full(cap, -1, dtype=np.int32)

        a_is_start = True  # which tree is currently Va (rooted at start)

        for _ in range(self.max_iter):
            # ── Extend Tree A toward a random sample biased to root of B ──
            q_rand   = self._sample(Vb[0])
            i_near_a = self._nearest(Va, na, q_rand)
            q_new    = self._steer(Va[i_near_a], q_rand)
            if q_new is None or not self._inside(q_new) or \
               not self._collision_free(Va[i_near_a], q_new):
                # Swap even on failure to keep growth balanced
                Va, Pa, na, Vb, Pb, nb, a_is_start = \
                    Vb, Pb, nb, Va, Pa, na, not a_is_start
                continue

            Va[na] = q_new;  Pa[na] = i_near_a;  i_new_a = na;  na += 1

            # ── Greedy-connect Tree B toward the new node ──
            i_near_b  = self._nearest(Vb, nb, q_new)
            q_cur     = Vb[i_near_b].copy()
            i_last_b  = i_near_b
            connected = False

            while True:
                q_next = self._steer(q_cur, q_new)
                if q_next is None or not self._collision_free(q_cur, q_next):
                    break
                Vb[nb] = q_next;  Pb[nb] = i_last_b;  i_last_b = nb;  nb += 1
                q_cur = q_next
                if np.linalg.norm(q_cur - q_new) < self.step * 0.5:
                    connected = True
                    break

            if connected:
                path_a = self._backtrack(Va, Pa, i_new_a)   # [q_new, …, root_a]
                path_b = self._backtrack(Vb, Pb, i_last_b)  # [q_connect, …, root_b]
                if a_is_start:
                    # root_a=start, root_b=goal → start→…→q_new + q_connect→…→goal
                    return np.vstack(path_a[::-1] + path_b)
                else:
                    # root_a=goal, root_b=start → start→…→q_connect + q_new→…→goal
                    return np.vstack(path_b[::-1] + path_a)

            # ── Swap trees for balanced bidirectional growth ──
            Va, Pa, na, Vb, Pb, nb, a_is_start = \
                Vb, Pb, nb, Va, Pa, na, not a_is_start

        return None  # No path found
