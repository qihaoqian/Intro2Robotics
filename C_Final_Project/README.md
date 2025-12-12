# 视觉伺服残差学习项目

使用残差神经网络改进IBVS视觉伺服控制，包含完整的baseline对比实验框架。

## 🎯 项目定位

**研究目标**: 通过残差学习改进传统视觉伺服性能

**方法对比**:
- **Baseline**: 传统IBVS控制 (v = IBVS(error))
- **Proposed**: IBVS + 残差NN (v = IBVS(error) + NN_residual)

**核心创新**: Ground Truth = 残差（v_actual - v_model），而非控制命令本身

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行baseline演示
python scripts/run_baseline.py

# 3. 收集训练数据（测试）
python scripts/collect_data.py --episodes 10

# 4. 训练模型
python scripts/train_model.py --epochs 50

# 5. 测试残差学习控制器
python scripts/run_residual_learning.py --model models/best_model.pth

# 6. 运行完整对比实验
python scripts/run_comparison.py --trials 5
```

### 🆕 相机速度残差模型（新）

```bash
# 训练相机速度残差模型（6维输出）
python scripts/train_camera_model.py \
    --data-dir data/residual_training_800_episodes \
    --epochs 500 \
    --batch-size 128 \
    --save-dir models/camera_residual

# 运行相机残差控制器
python scripts/run_camera_residual.py \
    --model models/camera_residual/best_camera_model.pth
```

详细说明见 `QUICK_START_CAMERA_RESIDUAL.md` 和 `CAMERA_RESIDUAL_MIGRATION.md`

## 项目结构

```
C_Final_Project/
├── src/                    # 源代码
│   ├── core/               # 核心功能（IBVS、机器人、相机）
│   ├── baseline/           # Baseline IBVS控制器
│   ├── learning/           # 残差学习（数据收集、模型）
│   └── utils/              # 工具函数
├── scripts/                # 可执行脚本
│   ├── run_baseline.py     # 运行baseline
│   ├── collect_data.py     # 收集数据
│   ├── train_model.py      # 训练模型
│   ├── run_residual_learning.py  # 运行残差学习控制器 ⭐
│   ├── run_comparison.py   # 完整对比实验 ⭐
│   └── visualize_data.py   # 数据可视化
├── docs/                   # 文档
│   ├── TRAINING_GUIDE.md   # 训练指南（GT定义）⭐
│   ├── PROJECT_STRUCTURE.md
│   └── API.md
├── data/                   # 训练数据（自动生成）
└── models/                 # 模型（自动生成）
```

## Ground Truth定义（重要）

### 核心问题

**训练时的Ground Truth是什么？**

### 答案

**GT = v_residual = v_actual - v_model**（残差修正量）

```python
# 时刻t
v_model = IBVS_controller(error_t)      # 传统方法预测

# 执行控制后
state_{t+1} = observe()
v_actual = (state_{t+1} - state_t) / dt # 实际速度

# Ground Truth
v_residual_GT = v_actual - v_model      # ← 训练标签
```

### 为什么这样定义

- ❌ 如果 GT = v_model → NN只能模仿IBVS，无法改进
- ✅ 如果 GT = v_residual → NN学习修正量，可以改进

**详细解释**: `docs/TRAINING_GUIDE.md`

## 使用方法

### 运行Baseline实验

```bash
python scripts/run_baseline.py --num-targets 5 --seed 42
```

观察传统IBVS的性能表现。

### 收集训练数据

```bash
# 快速测试（10 episodes）
python scripts/collect_data.py --episodes 10 --targets 3

# 标准收集（200 episodes，推荐）
python scripts/collect_data.py --episodes 200 --targets 5 --max-iters 300
```

**数据自动包含**:
- ✅ 连续时间步状态 (t, t+1)
- ✅ IBVS控制输出 (v_model)
- ✅ 自动计算的残差GT (v_actual - v_model)

数据保存在 `data/residual_training/`

### 验证数据质量

```bash
python src/learning/data_loader.py --data-dir data/residual_training
```

检查：
- 样本数 > 10,000 ✓
- 成功率 > 50% ✓
- 残差分布合理 ✓

### 训练模型

```bash
python scripts/train_model.py \
    --data-dir data/residual_training \
    --epochs 100 \
    --batch-size 64 \
    --lr 1e-3
```

模型保存: `models/best_model.pth`

训练监控:
```bash
tensorboard --logdir models/logs
```

### 测试残差学习控制器

```bash
# 运行残差学习控制器
python scripts/run_residual_learning.py --model models/best_model.pth

# 录制视频
python scripts/run_residual_learning.py --video

# 测试baseline模式（禁用残差）
python scripts/run_residual_learning.py --baseline
```

### 运行完整对比实验

```bash
# 标准对比（5次试验）
python scripts/run_comparison.py --trials 5

# 详细对比（10次试验）
python scripts/run_comparison.py --model models/best_model.pth --trials 10
```

输出：
- 控制台统计结果
- 可视化对比图 `comparison_results.png`
- 性能改进分析

## 核心算法

### Baseline: IBVS

```
图像雅可比: L = ∂s/∂v (2×6)
控制律: v = -λ L⁺ e
```

### Learning: Residual NN

```
传统控制: v_model = IBVS(error)
残差预测: v_residual = ResidualNN(state, v_model)
改进控制: v_improved = v_model + α·v_residual
```

## 预期性能

| 指标 | Baseline | Learning | 改进 |
|------|---------|----------|------|
| 收敛时间 | 150 iter | 100 iter | -33% |
| 最终误差 | 8 px | 4 px | -50% |
| 成功率 | 85% | 95% | +10% |

## 技术栈

- **仿真**: PyBullet
- **机器人**: Franka Panda (7-DOF)
- **相机**: RGB-D (512×512, FOV=120°)
- **控制**: IBVS (基于图像的视觉伺服)
- **学习**: Residual NN (PyTorch)

## 配置参数

主要参数在 `src/core/config.py`：

```python
# 控制增益
CONTROL_GAIN_TRANSLATION = 0.15
CONTROL_GAIN_ROTATION = 0.15

# 收敛阈值
CONVERGENCE_THRESHOLD = 10.0  # 像素

# 目标配置
NUM_TARGETS = 5
TARGET_AREA_X = (0.2, 0.8)
TARGET_AREA_Y = (0.3, 1.2)
```

## 文档导航

- **README.md** (本文件) - 项目主文档
- **QUICKSTART.md** - 5分钟快速上手
- **docs/TRAINING_GUIDE.md** - 训练指南和GT定义 ⭐
- **docs/PROJECT_STRUCTURE.md** - 项目结构详解
- **docs/API.md** - API文档
- **项目整理完成.md** - 重组说明

## 依赖安装

```bash
pip install -r requirements.txt
```

主要依赖:
- numpy, pybullet, opencv-python, scipy
- pytorch, h5py, tqdm, matplotlib, tensorboard

## 常见问题

### Q: Ground Truth到底是什么？

**A**: GT = v_actual - v_model（残差），通过观察连续时间步自动计算。详见 `docs/TRAINING_GUIDE.md`

### Q: 为什么不用v_model作为GT？

**A**: 用v_model作GT，NN只能模仿IBVS，无法改进。用残差作GT，NN可以学习修正量。

### Q: 需要多少训练数据？

**A**:
- 最少: 10,000 样本
- 推荐: 50,000 样本 (200 episodes)
- 最佳: 100,000+ 样本

### Q: 如何确保改进有效？

**A**: 运行对比实验 `python scripts/run_comparison.py`，对比baseline和learning方法的收敛时间、误差、成功率。

## 许可

本项目仅用于教育和研究目的。
