from utils import *
from collections import deque
import copy
from dataclasses import dataclass
import argparse
from pathlib import Path

DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
@dataclass(frozen=True, slots=True)
class S_a:
    x: int
    y: int
    d: tuple[int, int]   
    k: bool
    dm_open: int
    dm_lock: int

def env_to_state(env, door_positions) -> S_a:
    raw = env.unwrapped
    x, y = map(int, raw.agent_pos)
    d_vec = tuple(map(int, raw.dir_vec))
    has_key = raw.carrying is not None
    m_open = m_lock = 0
    for i, (dx,dy) in enumerate(door_positions):
        door = raw.grid.get(dx,dy)
        if door:
            if door.is_open:   m_open |= 1<<i
            if door.is_locked: m_lock |= 1<<i
    return S_a(x,y,d_vec,has_key,m_open,m_lock)

def state_to_env(env, s:S_a, door_positions):
    raw = env.unwrapped
    raw.agent_pos = np.array([s.x,s.y])
    raw.agent_dir = DIRS.index(s.d)
    raw.carrying  = Key('yellow') if s.k else None
    for i,(dx,dy) in enumerate(door_positions):
        door = raw.grid.get(dx,dy)
        door.is_open   = bool(s.dm_open>>i & 1)
        door.is_locked = bool(s.dm_lock>>i & 1)


def succ_reward_by_env(env, s: S_a, a: int, door_positions):
    env_cp = copy.deepcopy(env)               

    state_to_env(env_cp, s, door_positions)   

    cost, term = step(env_cp, a)            
    s2   = env_to_state(env_cp, door_positions)
    if s2 == s:
        cost += 10
    if term:
        cost -= 100
    return s2, -cost, term

# Enumerate all reachable states
def enumerate_states(s0, env, door_positions):
    Q = deque([s0])
    seen = {s0}
    term_states = set()
    while Q:
        s = Q.popleft()
        for a in range(5):
            s2, _, done = succ_reward_by_env(env, s, a, door_positions)
            if done and s2 not in term_states:
                term_states.add(s2)
                print(s2)
            if not done and s2 not in seen:
                seen.add(s2)
                Q.append(s2)
                # print(s2)
    return seen | term_states

# Value Iteration
def value_iteration(states, env, door_positions, gamma=0.9, eps=1e-6):
    V = {s: 0.0 for s in states}
    while True:
        delta = 0.0
        for s in states:
            q_vals = []
            for a in range(5):
                s2, r, done = succ_reward_by_env(env, s, a, door_positions)
                v_next = 100 if done else V.get(s2, 0.0)
                q_vals.append(r + gamma * v_next)
            v_new = max(q_vals)
            delta = max(delta, abs(v_new - V[s]))
            V[s]  = v_new
        if delta < eps:
            break

    pi = {}
    for s in states:
        best_a, best_q = None, -1e9
        for a in range(5):
            s2, r, done = succ_reward_by_env(env, s, a, door_positions)
            v_next = 100 if done else V.get(s2, 0.0)
            q = r + gamma * v_next
            if s2 not in V:
                V[s2] = 0.0
            if q > best_q:
                best_q, best_a = q, a
        pi[s] = best_a
    return pi

# Rollout a policy to get an action sequence
def rollout_policy(env, s0, pi, door_positions, max_steps=500):
    env_cp = copy.deepcopy(env)

    # Start from s0
    state_to_env(env_cp, s0, door_positions)
    traj, s = [], s0

    for _ in range(max_steps):
        a = pi[s]
        traj.append(a)
        _, term = step(env_cp, a)
        s = env_to_state(env_cp, door_positions)
        if term:
            break

    return traj

def doorkey_problemA(env, info):
    # Collect static map information
    vec = info["door_pos"]
    door_positions = [(int(vec[0]), int(vec[1]))]
    print("Door Positions: ", door_positions)
    
    vec = info["init_agent_dir"]              
    d0  = (int(vec[0]), int(vec[1]))         
    
    # Initialize state
    init_state = S_a(
        info["init_agent_pos"][0],
        info["init_agent_pos"][1],
        d0,             # direction
        False,          # has_key
        0,              # door_open
        1,              # door_lock
    )

    # Enumerate states → Value Iteration → Policy
    states = enumerate_states(init_state, env, door_positions)
    pi     = value_iteration(states, env, door_positions)

    # rollout
    optim_act_seq = rollout_policy(env, init_state, pi, door_positions)
    return optim_act_seq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env", nargs="?", help="solve single .env (Part A) if provided")
    args = ap.parse_args()
    known = [Path(args.env)] if args.env else sorted(Path("envs/known_envs").glob("*.env"))
    
    for p in known:
        env, info = load_env(p) 
        seq = doorkey_problemA(env, info)  
        print(f"{p} Optimal Action Sequence: ", seq)
        filename = p.name                  
        out_dir = Path("gif/known_envs")
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / filename.replace('.env', '.gif')  
        draw_gif_from_seq(seq, env, str(out_path))        

if __name__ == "__main__":
    main()