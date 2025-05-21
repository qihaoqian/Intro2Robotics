"""
State Encoding
--------------

(ax, ay)       Agent coordinates       [0,9]²  (excluding static walls)
adir           Agent direction id      {0:↑,1:→,2:↓,3:←}
key_idx        Key position index      {0,1,2,3}
                0-2 → KEY_POSITIONS[i]; 3 → already carrying
open0, open1   Whether the two doors are open  {0,1}
goal_idx       Goal position index     {0,1,2}


Immediate Reward
----------------
* -1 per step (hitting wall or staying in place)
* +100 for reaching any goal and terminate
"""
from __future__ import annotations
import pickle
from minigrid.envs.doorkey import DoorKeyEnv
from typing import List, Tuple
from utils import *
from gymnasium.envs.registration import register
from minigrid.envs.doorkey import DoorKeyEnv
import copy

GRID = 10
WALL_X = 5
DOOR0 = (5, 3)  # index 0
DOOR1 = (5, 7)  # index 1
KEY_POSITIONS = [(2, 2), (2, 3), (1, 6)]
GOAL_POSITIONS = [(6, 1), (7, 3), (6, 6)]
START_POS = (4, 8)
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
MF, TL, TR, PK, UD = range(5)
GAMMA = 0.8
STEP_COST = -1
GOAL_REWARD = 100

State = Tuple[int, int, int, int, int, int, int]  # ax, ay, adir, key_idx, open0, open1, goal_idx

class DoorKey10x10Env(DoorKeyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=10, **kwargs)

register(
    id='MiniGrid-DoorKey-10x10-v0',
    entry_point='__main__:DoorKey10x10Env'
)

def is_static_wall(x: int, y: int) -> bool:
    if x < 0 or x >= GRID or y < 0 or y >= GRID:
        return True
    return (x == WALL_X) and (y not in (DOOR0[1], DOOR1[1]))


def door_closed(x: int, y: int, open0: int, open1: int) -> bool:
    if (x, y) == DOOR0:
        return open0 == 0
    if (x, y) == DOOR1:
        return open1 == 0
    return False


def blocked(x: int, y: int, o0: int, o1: int) -> bool:
    return is_static_wall(x, y) or door_closed(x, y, o0, o1)


def next_state(s: State, a: int) -> Tuple[State, float, bool]:
    ax, ay, adir, kidx, o0, o1, goal_idx = s
    reward = STEP_COST
    done = False

    # ------------------------------------------------ MF
    if a == MF:
        dx, dy = DIRS[adir]
        nx, ny = ax + dx, ay + dy
        if not blocked(nx, ny, o0, o1):
            ax, ay = nx, ny

    # ------------------------------------------------ TL / TR
    elif a == TL:
        adir = (adir - 1) % 4
    elif a == TR:
        adir = (adir + 1) % 4

    # ------------------------------------------------ PK
    elif a == PK and kidx != 3:
        dx, dy = DIRS[adir]
        if (ax+dx, ay+dy) == KEY_POSITIONS[kidx]:
            kidx = 3  # pick up key

    # ------------------------------------------------ UD
    elif a == UD and kidx == 3:
        dx, dy = DIRS[adir]
        if (ax+dx, ay+dy) == DOOR0 and o0 == 0:
            o0 = 1
        elif (ax+dx, ay+dy) == DOOR1 and o1 == 0:
            o1 = 1

    # detect goal
    if (ax, ay) == GOAL_POSITIONS[goal_idx]:
        reward += GOAL_REWARD
        done = True

    return (ax, ay, adir, kidx, o0, o1, goal_idx), reward, done

# 生成全状态 & 转移表
ALL_STATES: List[State] = []
for ax in range(GRID):
    for ay in range(GRID):
        if is_static_wall(ax, ay):
            continue
        for adir in range(4):
            for kidx in range(4):
                for o0 in (0, 1):
                    for o1 in (0, 1):
                        for goal_idx in range(3):
                            
                            ALL_STATES.append((ax, ay, adir, kidx, o0, o1, goal_idx))
S_INDEX = {s: i for i, s in enumerate(ALL_STATES)}
N_S = len(ALL_STATES)

# precompute P
P = [[None] * 5 for _ in range(N_S)]
for s in ALL_STATES:
    sid = S_INDEX[s]
    for a in range(5):
        ns, r, done = next_state(s, a)
        P[sid][a] = (S_INDEX[ns], r, done)

# Value Iteration
V = [0.0] * N_S
EPS = 1e-6
while True:
    delta = 0.0
    for sid in range(N_S):
        q_best = max(P[sid][a][1] + GAMMA * (0.0 if P[sid][a][2] else V[P[sid][a][0]]) for a in range(5))
        delta = max(delta, abs(q_best - V[sid]))
        V[sid] = q_best
    if delta < EPS:
        break
print(f"[VI] Converged |S|={N_S}")

# extract policy
policy = [0] * N_S
for sid in range(N_S):
    best_a, best_q = 0, -1e9
    for a in range(5):
        nsid, r, done = P[sid][a]
        q = r + GAMMA * (0.0 if done else V[nsid])
        if q > best_q:
            best_q, best_a = q, a
    policy[sid] = best_a

with open("policy.pkl", "wb") as f:
    pickle.dump(policy, f)
print("[VI] Saved optimal policy to policy.pkl")

from utils import step, load_random_env  

def env_to_universal_state(env, info) -> State:
    raw = env.unwrapped
    ax, ay = map(int, raw.agent_pos)
    adir = DIRS.index(tuple(map(int, raw.dir_vec)))
    # door open status
    door0_open = int(raw.grid.get(*DOOR0).is_open)
    door1_open = int(raw.grid.get(*DOOR1).is_open)
    # key position index or 3
    if raw.carrying is not None:
        key_idx = 3
    else:
        kx, ky = map(int, info["key_pos"])
        key_idx = KEY_POSITIONS.index((kx, ky))
    
    goal_idx = GOAL_POSITIONS.index((int(info["goal_pos"][0]), int(info["goal_pos"][1])))
    return (ax, ay, adir, key_idx, door0_open, door1_open, goal_idx)


def rollout_in_env(env, info, max_steps: int = 100) -> Tuple[bool, List[int]]:
    seq: List[int] = []
    s = env_to_universal_state(env, info)
    for _ in range(max_steps):
        a = policy[S_INDEX[s]]
        seq.append(a)
        _, terminated = step(env, a)
        if terminated:
            return True, seq
        s = env_to_universal_state(env, info)
    return False, seq

def batch_test(env_folder: str, rounds: int = 100):
    success = 0
    for _ in range(rounds):
        env, info, _ = load_random_env(env_folder)
        env_cp = copy.deepcopy(env)
        ok, action_seq = rollout_in_env(env_cp, info)
        success += ok
        if ok:
            result = 'gif/' + _.replace('\\','/').split('envs/',1)[1][:-4] + '.gif'
            os.makedirs(os.path.dirname(result), exist_ok=True)
            draw_gif_from_seq(action_seq, env, result)
            print(f'{result}:', action_seq)
            
    print(f"Solved {success}/{rounds} randomly‑sampled maps using universal π.")

        
if __name__ == "__main__":
    env_folder = "./envs/random_envs"
    batch_test(env_folder)

