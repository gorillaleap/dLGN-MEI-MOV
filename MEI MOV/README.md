# dLGN-MEI: Biologically Interpretable Maximally Exciting Inputs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)

**A Video-Level MEI Generation Framework with Gaussian Factorized Readout and Three-Phase Anti-Adversarial Pipeline**

English | [简体中文](#-中文版)

---

## Table of Contents

- [1. Background & Problem](#1-background--problem)
- [2. Data Structure & Experimental Paradigm](#2-data-structure--experimental-paradigm)
- [3. Key Features](#3-key-features)
- [4. Directory Structure](#4-directory-structure)
- [5. Installation](#5-installation)
- [6. Usage](#6-usage)
- [7. Model Architecture](#7-model-architecture)
- [8. Three-Phase Pipeline](#8-three-phase-pipeline)
- [9. Key Findings & Results](#9-key-findings--results)
- [10. Citation](#10-citation)

---

## 1. Background & Problem

### What is MEI?

**Maximally Exciting Images (MEI)** are visual stimulus patterns that elicit the strongest response from a specific neuron. Through MEI analysis, we can reverse-engineer a neuron's feature selectivity and understand how the visual system encodes external information.

### The Problem with Traditional Methods

In conventional MEI generation pipelines, gradient-based optimization methods often cause models to converge to **high-frequency local optima** (adversarial attractors). This manifests as:

| Artifact Type | Visual Characteristics | Biological Implausibility |
|---------------|------------------------|---------------------------|
| Checkerboard | Periodic light-dark alternation | RGC receptive fields lack periodic structures |
| Salt-and-pepper noise | Random high-frequency pixels | Exceeds neuronal spatial resolution |
| Edge aliasing | Discontinuous sharp boundaries | Contradicts Gabor filter smoothness |

These **high-frequency aliasing artifacts** not only degrade MEI visual quality but, more importantly, **violate the receptive field properties of biological retinal ganglion cells (RGCs)** — real RGC receptive fields typically exhibit smooth Gabor wave or Gaussian envelope structures.

> **Core Contradiction**: Models possess the ability to "cheat" (capturing arbitrary high frequencies) but lack the "discipline" to follow biological priors.

---

## 2. Data Structure & Experimental Paradigm

The digital twin model in this project is built upon real neurobiological experimental data.

### 2.1 Biological Recording Methodology

We employed **two-photon calcium imaging** to record neural activity in the **dorsal Lateral Geniculate Nucleus (dLGN)** of **awake mice**:

| Recording Parameter | Details |
|---------------------|---------|
| **Imaging Technique** | Two-photon excitation fluorescence microscopy |
| **Target Brain Region** | Dorsal LGN (dLGN) — the relay station for visual information from retina to cortex |
| **Cell Type** | Cart-positive retinal ganglion cell (RGC) axon terminals (boutons) |
| **Calcium Indicator** | GCaMP series genetically-encoded calcium indicator |
| **Recording Target** | Presynaptic boutons formed by RGC axons in dLGN |

> **Biological Significance**: By recording RGC axon terminals rather than cell bodies, we directly measure the visual signals transmitted to dLGN, providing a unique perspective for understanding retino-thalamic information transfer.

### 2.2 Visual Stimulation Paradigm

```
Stimulus Design: 48 distinct visual stimulus patterns
Repetitions: Each stimulus presented 30 times (trials)
Total Stimuli: 48 × 30 = 1,440 presentations
```

### 2.3 Data Matrix Construction

From the extensive neural recordings, we rigorously selected **50 direction-selective (DS) boutons** to construct the core dataset:

| Matrix | Dimensions | Content |
|--------|------------|---------|
| **Stimulus** | [1440, H, W] | 48 stimuli × 30 repetitions |
| **Response** | [1440, 50] | Calcium fluorescence from 50 DS boutons |
| **Behavior** | [1440, 2] | Running speed + Pupil size |

### 2.4 Significance of Direction Selectivity

We specifically selected **direction-selective (DS) boutons** because:

1. **Clear Function**: DS neurons produce strongest responses to specific motion directions
2. **Interpretable Features**: MEI analysis can clearly reveal orientation and direction preferences
3. **Model Validation**: DS properties provide objective criteria for evaluating model accuracy

---

## 3. Key Features

- 🧠 **Gaussian Factorized Readout** — 99.8% parameter reduction vs. fully-connected layers
- 🔬 **Three-Phase Anti-Adversarial Pipeline** — Natural search → Parametric seed → Pixel optimization
- 🎥 **Video-Level MEI** — 33-frame spatiotemporally continuous stimuli
- 🔍 **Feature Collapse Detection** — `check_weights.py` diagnostic tool

---

## 4. Directory Structure

```
dLGN-MEI/
├── mei-test/                              # Data Preprocessing & Signal Conversion
│   ├── environment.yml                    # Conda env: cascade
│   ├── convert.py                         # dF/F → Spike conversion
│   ├── cascade2p/                         # CASCADE neural network inference
│   │   ├── cascade.py
│   │   ├── utils.py
│   │   └── config.py
���   └── Pretrained_models/
│       └── Global_EXC_15Hz_smoothing100ms/
│
└── mei-mov/                               # MEI Generation Pipeline
    ├── environment.yml                    # Conda env: chatgpt
    ├── model.py                           # DeepRetina3D + Gaussian Factorized Readout
    ├── dataset.py                         # Data loader
    ├── train.py                           # Training core logic
    ├── run_training.py                    # One-click training script
    ├── generate_three_phase_comparison.py # Three-phase MEI generation
    ├── generate_informed_mei.py           # Two-phase MEI (testing)
    └── check_weights.py                   # Feature collapse detection
```

---

## 5. Installation

### Two Independent Conda Environments

```bash
# 1. Clone the repository
git clone https://github.com/gorillaleap/dLGN-MEI.git
cd dLGN-MEI

# 2. Create mei-test environment (Calcium signal conversion)
cd mei-test
conda env create -f environment.yml
conda activate cascade

# 3. Create mei-mov environment (Model training + MEI generation)
cd ../mei-mov
conda env create -f environment.yml
conda activate chatgpt
```

### Environment Overview

| Folder | Environment Name | Python | Key Dependencies |
|--------|------------------|--------|------------------|
| mei-test | `cascade` | 3.9 | cascadetorch, h5py |
| mei-mov | `chatgpt` | 3.11 | pytorch, swanlab |

---

## 6. Usage

### Step 1: Calcium → Spikes (mei-test)

Convert ΔF/F calcium signals to spike rates using CASCADE:

```bash
conda activate cascade
cd mei-test
python convert.py
```

**Output**: `cascade_spikes_23frames.npy` (23, 1440, 50)

**Features**:
- Loads 85-frame raw ΔF/F signals from `.mat` file
- Mirror padding (PAD_WIDTH=60) for edge handling
- CASCADE neural network inference (GPU accelerated)
- Extracts stimulus window [32:55] → 23 frames
- Generates `spike_check.png` verification plot

### Step 2: Train Digital Twin Model (mei-mov)

```bash
conda activate chatgpt
cd mei-mov
python run_training.py
```

**Output**: `checkpoints/best_model.pth`

**Configuration**:
- Batch size: 52
- Epochs: 25
- Learning rate: 5e-5
- Optimizer: Adam (optional SAM)
- Loss: MSE + L1 Sparsity - Pearson Correlation

### Step 3: Check for Feature Collapse

```bash
python check_weights.py
```

Detects whether neurons have learned identical features by computing the correlation matrix of `feature_weights`.

**Output**: `weight_correlation.png` heatmap

**Interpretation**:
- Mean off-diagonal correlation > 0.9: Severe feature collapse
- Mean off-diagonal correlation > 0.6: High similarity
- Mean off-diagonal correlation < 0.6: Healthy diversity

### Step 4: Generate MEI

```bash
python generate_three_phase_comparison.py
```

Generates three-phase comparison for each neuron:
- `original_strongest.png/mp4` — Strongest natural stimulus
- `best_grating_seed.png/mp4` — Optimal parametric grating
- `final_mei.png/mp4` — Final optimized MEI

---

## 7. Model Architecture

### DeepRetina3D with Gaussian Factorized Readout

```
Input: (B, 1, 33, 80, 80) — 33-frame video stimulus
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Conv3d(1→32, k=5×7×7) → BN → Softplus → MaxPool        │
│  Output: (B, 32, 29, 40, 40)                            │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Conv3d(32→64, k=5×5×5) → BN → Softplus → MaxPool       │
│  Output: (B, 64, 25, 20, 20)                            │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Conv3d(64→64, k=3×5×5) → BN → Softplus → MaxPool       │
│  + Residual Connection                                   │
│  Output: (B, 64, 23, 10, 10)                            │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Gaussian Factorized Readout                             │
│  ├── mu: (N_neurons, 2) — RF center coordinates         │
│  ├── sigma: (N_neurons, 1) — RF size                    │
│  ├── feature_weights: (N_neurons, 64) — Feature prefs   │
│  └── bias: (N_neurons,) — Initialized to -2.0           │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  FiLM: Feature-wise Linear Modulation from behavior      │
│  Output: (B, 23, N_neurons) — Per-frame firing rate      │
└─────────────────────────────────────────────────────────┘
```

### Input/Output Specifications

| Component | Shape | Description |
|-----------|-------|-------------|
| Stimulus | (B, 1, 33, 80, 80) | 33-frame video stimulus |
| Behavior | (B, T, 2) | Running speed + Pupil size (zeros for neutral) |
| Output | (B, 23, N_neurons) | Per-frame firing rate (non-negative via Softplus) |

---

## 8. Three-Phase Pipeline

### Phase 0: Natural Search
- Searches real dataset for strongest natural stimulus
- Provides biological baseline for comparison
- Optional (skipped if no data provided)

### Phase 1: Parametric Seed
- 2D grid search: 8 directions × 3 spatial frequencies = 24 combinations
- Finds optimal Gabor grating parameters
- Locks in direction preference (θ) and spatial frequency (sf)

### Phase 2: Pixel Optimization
- Uses optimal grating as initialization seed
- Low learning rate (0.003) to preserve grating structure
- Gaussian blur + gradient mask regularization
- TV Loss for spatial/temporal smoothness

---

## 9. Key Findings & Results

### Complete Elimination of High-Frequency Artifacts

| Comparison | Traditional Methods | Our Method |
|------------|---------------------|------------|
| Checkerboard artifacts | Severe | **Completely eliminated** |
| Salt-and-pepper noise | Noticeable | **Completely eliminated** |
| Edge aliasing | Rough | **Smooth transitions** |
| Overall texture | Digital noise | **Silky smooth** |

### Biologically Plausible MEI Structures

Generated MEIs exhibit features consistent with RGC receptive field properties:
- **Gabor waves**: Smooth sinusoidal modulation structures
- **Gaussian envelopes**: Spatial decay with strong center, weak periphery
- **Orientation selectivity**: Clear orientation preferences

### Quantitative Results

```
Seeded MEI Average Enhancement: 1.5x ~ 2.0x
Random MEI Average Enhancement: 1.3x ~ 1.8x
Maximum Single Neuron Enhancement: > 3.0x
```

---

## 10. Citation

If you use this project in your research, please cite:

```bibtex
@misc{dlgn_mei_2026,
  title={dLGN-MEI: Biologically Interpretable Maximally Exciting Inputs with Gaussian Factorized Readout},
  author={dLGN-MEI Team},
  year={2026},
  howpublished={\url{https://github.com/gorillaleap/dLGN-MEI}}
}
```

### Acknowledgments

- **inception_loop** - Walker et al. (2019) Nature Neuroscience
- **CASCADE** - Rupprecht et al. (2021) eLife

---

<p align="center">
  <b>From High-Frequency Noise to Silky Smooth — A Biologically-Inspired MEI Architecture</b>
</p>

---

---

# 🇨🇳 中文版

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)

**基于高斯分解读出层与三阶段抗对抗管线的视频级 MEI 生成框架**

[English](#dgn-mei-biologically-interpretable-maximally-exciting-inputs) | 简体中文

---

## 目录

- [1. 背景与问题](#1-背景与问题)
- [2. 数据结构与实验范式](#2-数据结构与实验范式)
- [3. 核心特性](#3-核心特性)
- [4. 目录结构](#4-目录结构)
- [5. 安装指南](#5-安装指南)
- [6. 使用方法](#6-使用方法)
- [7. 模型架构](#7-模型架构)
- [8. 三阶段管线](#8-三阶段管线)
- [9. 主要研究成果](#9-主要研究成果)
- [10. 引用](#10-引用)

---

## 1. 背景与问题

### 什么是 MEI？

**最大兴奋图像（Maximally Exciting Image, MEI）** 是指能够使特定神经元产生最大响应的视觉刺激模式。通过 MEI 分析，我们可以逆向解码神经元的特征选择性，理解视觉系统如何编码外界信息。

### 传统方法的痛点

在传统的 MEI 生成流程中，基于梯度优化的方法往往会导致模型陷入**高频局部最优解**（对抗吸引子）。具体表现为：

| 伪影类型 | 视觉特征 | 生物学不合理性 |
|---------|---------|---------------|
| 棋盘格 | 周期性明暗交替 | RGC 感受野不具有周期性结构 |
| 麻点噪点 | 随机高频像素 | 超越了神经元的空间分辨率 |
| 边缘锯齿 | 不连续的锐利边界 | 与 Gabor 滤波器的平滑特性相悖 |

这些**高频锯齿伪影**不仅影响 MEI 的视觉质量，更重要的是**不符合生物视网膜神经节细胞（RGC）的感受野特性**——真实的 RGC 感受野通常呈现为平滑的 Gabor 波纹或高斯包络结构。

> **核心矛盾**：模型拥有"作弊"的能力（捕捉任意高频），却缺乏"自律"的约束（遵循生物学先验）。

---

## 2. 数据结构与实验范式

本项目的数字孪生模型基于真实的神经生物学实验数据构建。

### 2.1 生物学记录手段

我们采用**双光子钙成像技术**，在**清醒小鼠**的**背侧外膝体（dLGN）**中进行神经活动记录：

| 记录参数 | 详细���明 |
|---------|---------|
| **成像技术** | 双光子激发荧光显微镜 |
| **靶脑区** | 背侧外膝体（dLGN）——视觉信息从视网膜传递至皮层的中继站 |
| **细胞类型** | Cart-positive 视网膜神经节细胞（RGC）的轴突终末 |
| **钙指示剂** | GCaMP 系列基因编码钙指示剂 |
| **记录对象** | RGC 轴突在 dLGN 中形成的突触前终末 |

> **生物学意义**：通过记录 RGC 轴突终末而非细胞体，我们直接测量了传入 dLGN 的视觉信号，为理解视网膜-丘脑信息传递提供了独特视角。

### 2.2 视觉刺激范式

```
刺激设计：48 种视觉刺激模式
重复次数：每种刺激重复呈现 30 次
总刺激数：48 × 30 = 1,440 次呈现
```

### 2.3 数据矩阵构建

从海量神经记录数据中，我们严格筛选出 **50 个具有方向选择性（DS）** 的 boutons：

| 矩阵 | 维度 | 内容 |
|------|------|------|
| **Stimulus** | [1440, H, W] | 48 刺激 × 30 重复 |
| **Response** | [1440, 50] | 50 个 DS boutons 的钙荧光响应 |
| **Behavior** | [1440, 2] | 跑步速度 + 瞳孔大小 |

### 2.4 方向选择性的意义

我们特别选择**方向选择性（DS）boutons**作为研究对象：

1. **功能明确**：DS 神经元对特定运动方向产生最强响应
2. **特征可解释**：MEI 分析能够清晰揭示其朝向和方向偏好
3. **模型验证**：DS 特性为评估数字孪生模型的准确性提供了客观标准

---

## 3. 核心特性

- 🧠 **高斯分解读出层 (Gaussian Factorized Readout)** — 相比全连接层参数减少 99.8%
- 🔬 **三阶段抗对抗管线** — 自然搜索 → 参数化种子 → 像素优化
- 🎥 **视频级 MEI** — 33 帧时空连续刺激
- 🔍 **特征坍缩检测** — `check_weights.py` 诊断工具

---

## 4. 目录结构

```
dLGN-MEI/
├── mei-test/                              # 数据预处理与信号转换
│   ├── environment.yml                    # Conda 环境: cascade
│   ├── convert.py                         # dF/F → Spike 转换
│   ├── cascade2p/                         # CASCADE 神经网络推断引擎
│   │   ├── cascade.py
│   │   ├── utils.py
│   │   └── config.py
│   └── Pretrained_models/
│       └── Global_EXC_15Hz_smoothing100ms/
│
└── mei-mov/                               # MEI 生成管线
    ├── environment.yml                    # Conda 环境: chatgpt
    ├── model.py                           # DeepRetina3D + 高斯分解读出层
    ├── dataset.py                         # 数据加载器
    ├── train.py                           # 训练核心逻辑
    ├── run_training.py                    # ���键训练脚本
    ├── generate_three_phase_comparison.py # 三阶段 MEI 生成
    ├── generate_informed_mei.py           # 两阶段 MEI (测试用)
    └── check_weights.py                   # 特征坍缩检测
```

---

## 5. 安装指南

### 两个独立 Conda 环境

```bash
# 1. 克隆仓库
git clone https://github.com/gorillaleap/dLGN-MEI.git
cd dLGN-MEI

# 2. 创建 mei-test 环境 (钙信号转换)
cd mei-test
conda env create -f environment.yml
conda activate cascade

# 3. 创建 mei-mov 环境 (模型训练 + MEI 生成)
cd ../mei-mov
conda env create -f environment.yml
conda activate chatgpt
```

### 环境说明

| 文件夹 | 环境名 | Python | 主要依赖 |
|--------|--------|--------|----------|
| mei-test | `cascade` | 3.9 | cascadetorch, h5py |
| mei-mov | `chatgpt` | 3.11 | pytorch, swanlab |

---

## 6. 使用方法

### Step 1: 钙信号 → 尖峰 (mei-test)

使用 CASCADE 将 ΔF/F 钙信号转换为放电率：

```bash
conda activate cascade
cd mei-test
python convert.py
```

**输出**: `cascade_spikes_23frames.npy` (23, 1440, 50)

**功能**:
- 从 `.mat` 文件加载 85 帧原始 ΔF/F 信号
- 镜像填充 (PAD_WIDTH=60) 处理边缘
- CASCADE 神经网络推断 (GPU 加速)
- 截取刺激窗口 [32:55] → 23 帧
- 生成 `spike_check.png` 校验图

### Step 2: 训练数字孪生模型 (mei-mov)

```bash
conda activate chatgpt
cd mei-mov
python run_training.py
```

**输出**: `checkpoints/best_model.pth`

**配置**:
- 批大小: 52
- 轮数: 25
- 学习率: 5e-5
- 优化器: Adam (可选 SAM)
- 损失: MSE + L1 Sparsity - Pearson Correlation

### Step 3: 检测特征坍缩

```bash
python check_weights.py
```

通过计算 `feature_weights` 的相关系数矩阵，检测神经元是否学习到相同特征。

**输出**: `weight_correlation.png` 热力图

**判断标准**:
- 平均非对角线相关性 > 0.9: 极其严重的特征坍缩
- 平均非对角线相关性 > 0.6: 权重高度相似
- 平均非对角线相关性 < 0.6: 健康

### Step 4: 生成 MEI

```bash
python generate_three_phase_comparison.py
```

为每个神经元生成三阶段对比：
- `original_strongest.png/mp4` — 最强自然刺激
- `best_grating_seed.png/mp4` — 最优参数化光栅
- `final_mei.png/mp4` — 最终优化 MEI

---

## 7. 模型架构

### DeepRetina3D + 高斯分解读出层

```
输入: (B, 1, 33, 80, 80) — 33 帧视频刺激
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Conv3d(1→32, k=5×7×7) → BN → Softplus → MaxPool        │
│  输出: (B, 32, 29, 40, 40)                               │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Conv3d(32→64, k=5×5×5) → BN → Softplus → MaxPool       │
│  输出: (B, 64, 25, 20, 20)                               │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Conv3d(64→64, k=3×5×5) → BN → Softplus → MaxPool       │
│  + 残差连接                                              │
│  输出: (B, 64, 23, 10, 10)                               │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  高斯分解读出层 (Gaussian Factorized Readout)            │
│  ├── mu: (N_neurons, 2) — RF 中心坐标                    │
│  ├── sigma: (N_neurons, 1) — RF 大小                     │
│  ├── feature_weights: (N_neurons, 64) — 特征偏好         │
│  └── bias: (N_neurons,) — 初始化为 -2.0                  │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  FiLM: 行为信号的特征线性调制                             │
│  输出: (B, 23, N_neurons) — 每帧放电率                   │
└─────────────────────────────────────────────────────────┘
```

### 输入/输出规格

| 组件 | 形状 | 说明 |
|------|------|------|
| Stimulus | (B, 1, 33, 80, 80) | 33 帧视觉刺激 |
| Behavior | (B, T, 2) | 跑步��度 + 瞳孔大小 (全零表示中性状态) |
| Output | (B, 23, N_neurons) | 每帧放电率 (Softplus 保证非负) |

---

## 8. 三阶段管线

### Phase 0: 自然搜索
- 遍历真实数据集寻找最强自然刺激
- 提供生物学基线对比
- 可选 (无数据时跳过)

### Phase 1: 参数化种子
- 2D 网格搜索: 8 方向 × 3 空间频率 = 24 种组合
- 锁定最优 Gabor 光栅参数
- 确定方向偏好 (θ) 和空间频率 (sf)

### Phase 2: 像素优化
- 使用最优光栅作为初始化种子
- 低学习率 (0.003) 保持光栅结构
- 高斯模糊 + 梯度掩码正则化
- TV Loss 保证时空平滑性

---

## 9. 主要研究成果

### 高频伪影的彻底消除

| 对比维度 | 传统方法 | 本项目方法 |
|---------|---------|-----------|
| 棋盘格伪影 | 严重 | **完全消除** |
| 麻点噪点 | 明显 | **完全消除** |
| 边缘锯齿 | 粗糙 | **平滑过渡** |
| 整体质感 | 数字噪声 | **丝绸般平滑** |

### 生物学合理的 MEI 结构

生成的 MEI 呈现出符合 RGC 感受野特性的特征：
- **Gabor 波纹**：平滑的正弦调制结构
- **高斯包络**：中心强、边缘弱的空间衰减
- **朝向选择性**：清晰的朝向偏好

### 量化结果

```
Seeded MEI 平均提升率: 1.5x ~ 2.0x
Random MEI 平均提升率: 1.3x ~ 1.8x
最大单神经元提升: > 3.0x
```

---

## 10. 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@misc{dlgn_mei_2026,
  title={dLGN-MEI: 基于高斯分解读出层的生物学可解释最大兴奋图像生成},
  author={dLGN-MEI Team},
  year={2026},
  howpublished={\url{https://github.com/gorillaleap/dLGN-MEI}}
}
```

### 致谢

- **inception_loop** - Walker et al. (2019) Nature Neuroscience
- **CASCADE** - Rupprecht et al. (2021) eLife

---

<p align="center">
  <b>从高频噪声到丝绸般平滑 —— 生物学启发的 MEI 生成架构</b>
</p>