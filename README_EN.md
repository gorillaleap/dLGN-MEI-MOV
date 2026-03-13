# dLGN-MEI: Biologically Interpretable Maximally Exciting Inputs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)

**A Video-Level MEI Generation Framework with Gaussian Factorized Readout and Three-Phase Anti-Adversarial Pipeline**

English | [简体中文](README.md)

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
│   └── Pretrained_models/
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
└─────────────────────────────────────��───────────────────┘
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