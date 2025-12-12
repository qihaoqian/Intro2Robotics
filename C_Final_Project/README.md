# Visual Servoing with Residual Reinforcement Learning

This project implements Image-Based Visual Servoing (IBVS) enhanced with residual reinforcement learning using PPO.

## Overview

- **Baseline**: Traditional IBVS controller
- **Proposed**: IBVS + Residual RL (PPO learns a multiplicative correction to IBVS commands)

## Project Structure

```
C_Final_Project/
├── scripts/
│   ├── run_baseline.py      # Baseline IBVS controller
│   └── run_residual_rl.py   # Residual RL training & evaluation
├── utils/
│   ├── config.py            # Configuration parameters
│   ├── robot.py             # Franka Panda robot class
│   ├── simulation.py        # PyBullet environment setup
│   ├── camera_utils.py      # Camera utilities
│   ├── controllers.py       # IBVS controller implementation
│   ├── target_selection.py  # Target detection and selection
│   └── visualization.py     # Visualization tools
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run Baseline IBVS

```bash
# Run N episodes of baseline visual servoing
python scripts/run_baseline.py -n 10

# With GUI visualization
python scripts/run_baseline.py -n 5 --gui
```

### Train Residual RL

```bash
# Train PPO residual policy
python scripts/run_residual_rl.py --train --timesteps 50000

# Resume training from checkpoint
python scripts/run_residual_rl.py --train --model-load-path ppo_residual_servo.zip --timesteps 20000

# With GUI during training
python scripts/run_residual_rl.py --train --gui --timesteps 10000
```

### Evaluate Trained Model

```bash
# Evaluate trained policy
python scripts/run_residual_rl.py --eval --model-load-path ppo_residual_servo.zip --episodes 10 --gui
```

## Method

The residual RL agent learns a multiplicative coefficient to adjust IBVS commands:

```
v_cmd = v_ibvs * (1 + 0.3 * tanh(action))
```

**Observation space** (23-dim):
- Visual error (normalized pixel coordinates)
- Depth at target
- Joint angles & velocities
- IBVS command (delta_X, delta_Omega)

**Action space** (6-dim):
- Residual coefficients for [δX, δΩ]

**Reward**:
- Tracking error penalty
- Energy cost (joint velocity)
- Step penalty
- Failure penalty (target lost)

## Configuration

Key parameters in `utils/config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `CAMERA_WIDTH/HEIGHT` | 512 | Image resolution |
| `CAMERA_FOV` | 120° | Field of view |
| `NUM_TARGETS` | 5 | Number of random targets |
| `MAX_ITERATIONS` | 100 | Max steps per episode |
| `CONVERGENCE_THRESHOLD` | 3 px | Success threshold |

## Tech Stack

- **Simulation**: PyBullet
- **Robot**: Franka Panda (7-DOF) with eye-in-hand camera
- **RL**: PPO (Stable-Baselines3)
- **Framework**: PyTorch, OpenCV, NumPy

## License

For educational and research purposes only.
