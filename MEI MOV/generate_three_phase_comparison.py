#!C:/Users/vipuser/anaconda3/envs/chatgpt/python.exe
# -*- coding: utf-8 -*-
"""
三维对比组合生成脚本 - Three-Phase Comparison Pipeline

这套流程用于顶级会议/汇报的"三维对比组合":
    Phase 0: 真实数据集最强刺激检索 (可选)
    Phase 1: 最佳光栅种子生成 (2D Grid Search)
    Phase 2: 最终 Informed MEI 优化

对每个神经元，在其专属文件夹内保存三种视觉刺激的对比图:
    - Original Strongest (真实数据) - 可选
    - Best Grating Seed (参数化光栅)
    - Final Informed MEI (优化结果)

配置说明:
    - 修改 DATA_MAT_PATH 指向你的 .mat 数据文件以启用 Phase 0
    - 如果没有数据文件，Phase 0 会自动跳过，只运行 Phase 1 + Phase 2

模型规格:
    - Stimulus 输入: (B, 1, T, 80, 80) - T 可为 23 (训练数据) 或 33 (MEI 生成)
    - Behavior 输入: (B, T, 2) - 固定为全0 (Neutral State)
    - 模型输出: (B, T, N_neurons) - Firing Rate

Author: Claude
Date: 2025-03
"""

import sys
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from torch.utils.data import Dataset, DataLoader
import h5py

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import create_model


# ============================================================
# Phase 0: Stimulus-Only DataLoader (无需 behavior)
# ============================================================

class StimulusOnlyDataset(Dataset):
    """
    简化数据集 - 仅加载 Stimulus，用于 Phase 0 搜索最强自然刺激

    与训练数据集的区别:
        - 只返回 stimulus，不需要 behavior/response
        - behavior 固定为全 0 (Neutral State)
        - 适合 MEI 生成场景
    """
    def __init__(
        self,
        mat_path: str,
        rf_diameter: int = 80,
        rf_center: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
        expand_factor: int = 30,
        time_frames: int = 23  # 与训练数据一致
    ):
        self.rf_diameter = rf_diameter
        self.rf_radius = rf_diameter // 2
        self.normalize = normalize
        self.time_frames = time_frames

        # 加载 stimulus
        print(f"\n[StimulusOnlyDataset] Loading from: {mat_path}")
        with h5py.File(mat_path, 'r') as f:
            stim_raw = f['stimulus'][:].T  # (Trials, Time, H, W)
            print(f"  Raw stimulus shape: {stim_raw.shape}")

        self.n_base_trials, self.time_frames, self.full_height, self.full_width = stim_raw.shape
        self.n_samples = self.n_base_trials * expand_factor

        # RF 中心
        if rf_center is not None:
            self.rf_center_h, self.rf_center_w = rf_center
        else:
            self.rf_center_h = self.full_height // 2
            self.rf_center_w = self.full_width // 2

        # 归一化参数
        if normalize:
            self.stim_mean = stim_raw.mean()
            self.stim_std = stim_raw.std()
        else:
            self.stim_mean = 0.0
            self.stim_std = 1.0

        # 预处理所有 stimulus
        self.all_stimuli = self._preprocess_stimuli(stim_raw, expand_factor)
        print(f"  Preprocessed stimulus shape: {self.all_stimuli.shape}")
        print(f"  Total samples: {self.n_samples}")

    def _preprocess_stimuli(self, stim_raw: np.ndarray, expand_factor: int) -> torch.Tensor:
        """预处理: RF裁剪 + 归一化"""
        n_base, T, H, W = stim_raw.shape
        n_total = n_base * expand_factor

        all_stimuli = torch.zeros(
            n_total, 1, T, self.rf_diameter, self.rf_diameter,
            dtype=torch.float32
        )

        center_h = self.rf_center_h
        center_w = self.rf_center_w

        for i in range(n_total):
            base_idx = i // expand_factor
            stim = stim_raw[base_idx]

            # RF 裁剪
            start_h = center_h - self.rf_radius
            start_w = center_w - self.rf_radius
            end_h = center_h + self.rf_radius
            end_w = center_w + self.rf_radius

            # 边界处理
            pad_h_before = max(0, -start_h)
            pad_w_before = max(0, -start_w)
            pad_h_after = max(0, end_h - H)
            pad_w_after = max(0, end_w - W)

            actual_start_h = max(0, start_h)
            actual_start_w = max(0, start_w)
            actual_end_h = min(H, end_h)
            actual_end_w = min(W, end_w)

            stim_crop = stim[:, actual_start_h:actual_end_h, actual_start_w:actual_end_w]

            if pad_h_before > 0 or pad_h_after > 0 or pad_w_before > 0 or pad_w_after > 0:
                stim_crop = np.pad(
                    stim_crop,
                    ((0, 0), (pad_h_before, pad_h_after), (pad_w_before, pad_w_after)),
                    mode='constant', constant_values=0
                )

            # 归一化
            stim_crop = (stim_crop - self.stim_mean) / (self.stim_std + 1e-8)
            all_stimuli[i, 0] = torch.from_numpy(stim_crop)

        return all_stimuli

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with:
                stimulus: (1, T, 80, 80)
                behavior: (T, 2) - 全零
        """
        return {
            'stimulus': self.all_stimuli[idx],
            'behavior': torch.zeros(self.time_frames, 2)
        }


def create_stimulus_dataloader(
    mat_path: str,
    batch_size: int = 32,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    """
    创建 Phase 0 专用 DataLoader

    Args:
        mat_path: .mat 数据文件路径
        batch_size: 批大小
        num_workers: DataLoader 工作进程数

    Returns:
        DataLoader: 返回 {'stimulus': ..., 'behavior': ...} 字典
    """
    dataset = StimulusOnlyDataset(mat_path, **kwargs)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # 不打乱，保持可复现
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


# ============================================================
# Phase 0: 真实数据集最强刺激检索
# ============================================================

def find_strongest_natural_stimulus(
    model: nn.Module,
    dataloader,
    neuron_idx: int,
    device: torch.device = torch.device('cpu')
) -> Tuple[torch.Tensor, float]:
    """
    Phase 0: 在真实数据集中搜索最强刺激

    遍历数据集，找出导致目标神经元产生最高峰值放电率的视频片段。
    这为三维对比组合提供"真实基准"。

    Args:
        model: 训练好的 DeepRetina3D 模型
        dataloader: StimulusOnlyDataset 的 DataLoader
        neuron_idx: 目标神经元索引
        device: 计算设备

    Returns:
        strongest_stimulus: 最强自然刺激 (1, 1, T, 80, 80), T=数据集帧数
        peak_response: 峰值响应值
    """
    model.eval()
    best_stimulus = None
    peak_response = -float('inf')

    print(f"\n{'='*60}")
    print(f"[Phase 0] Searching Strongest Natural Stimulus for Neuron {neuron_idx}")
    print(f"{'='*60}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            stimulus = batch['stimulus'].to(device)  # (B, 1, T, 80, 80)
            behavior = batch['behavior'].to(device)  # (B, T, 2)

            output = model(stimulus, behavior)  # (B, T, N_neurons)

            # 找峰值响应
            neuron_output = output[:, :, neuron_idx]  # (B, T)
            batch_peak = neuron_output.max(dim=1).values  # (B,)

            max_in_batch = batch_peak.max()
            if max_in_batch > peak_response:
                peak_response = max_in_batch
                max_idx = batch_peak.argmax()
                best_stimulus = stimulus[max_idx:max_idx+1].clone()
                print(f"  Found new peak: {peak_response:.4f} (batch {batch_idx})")

    if best_stimulus is None:
        # 如果没找到,生成随机噪声作为占位符
        print(f"  [Warning] No data found, using random noise as placeholder")
        # 获取时间帧数 (从 dataloader 的第一个 batch)
        for batch in dataloader:
            time_frames = batch['stimulus'].shape[2]
            break
        best_stimulus = torch.randn(1, 1, time_frames, 80, 80, device=device) * 0.1
        peak_response = 0.0

    print(f"\n  ★ [Phase 0 Found] Neuron {neuron_idx} Peak Response: {peak_response:.4f}")
    print(f"{'='*60}\n")

    return best_stimulus, peak_response.item()


# ============================================================
# Phase 1: 虚拟电生理测向 (方向 × 空间频率 2D Grid Search)
# ============================================================

def create_drifting_grating(
    direction: float,
    sf: float = 2.0,
    tf: float = 2.0,
    amplitude: float = 0.8,
    n_frames: int = 33,
    size: int = 80,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    创建完美滑动正弦光栅

    Args:
        direction: 方向 (度), 0° = 向右移动
        sf: 空间频率 (cycles/degree)
        tf: 时间频率 (Hz)
        amplitude: 振幅
        n_frames: 帧数
        size: 空间尺寸
        device: 计算设备

    Returns:
        grating: (1, 1, 33, 80, 80) 张量
    """
    # 创建空间坐标网格
    x = torch.linspace(-1, 1, size, device=device)
    y = torch.linspace(-1, 1, size, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    # 方向转换为弧度
    theta = np.deg2rad(direction)

    # 空间相位: 沿方向轴的投影
    spatial_phase = X * np.cos(theta) + Y * np.sin(theta)

    # 时间相位 (假设 30 FPS)
    t = torch.linspace(0, n_frames / 30, n_frames, device=device)

    # 生成光栅: sin(2π * sf * spatial - 2π * tf * t)
    grating = torch.zeros(1, 1, n_frames, size, size, device=device)
    for i, ti in enumerate(t):
        phase = 2 * np.pi * sf * spatial_phase - 2 * np.pi * tf * ti
        grating[0, 0, i] = amplitude * torch.sin(phase)

    return grating


def find_optimal_grating(
    model: nn.Module,
    neuron_idx: int,
    thetas: List[float] = None,
    sfs: List[float] = None,
    tf: float = 2.0,
    amplitude: float = 0.8,
    device: torch.device = torch.device('cpu')
) -> Tuple[torch.Tensor, float, float, float]:
    """
    Phase 1: Direction × Spatial Frequency 2D Grid Search

    扫描 8 方向 × 3 空间频率 = 24 种组合，找出全局最优。
    这减轻了 Phase 2 优化的负担，获得更精准的先验。

    Args:
        model: 训练好的 DeepRetina3D 模型
        neuron_idx: 目标神经元索引
        thetas: 测试方向列表 (度), 默认 8 个方向
        sfs: 空间频率列表, 默认 [1.5, 2.5, 3.5] (粗、中、细)
        tf: 时间频率 (固定 2.0)
        amplitude: 光栅振幅 (固定 0.8)
        device: 计算设备

    Returns:
        optimal_grating: 最佳光栅 (1, 1, 33, 80, 80)
        best_theta: 最佳方向 (度)
        best_sf: 最佳空间频率
        best_response: 最佳响应值
    """
    if thetas is None:
        thetas = [0, 45, 90, 135, 180, 225, 270, 315]
    if sfs is None:
        sfs = [1.5, 2.5, 3.5]  # 粗、中、细三种条纹

    model.eval()
    behavior = torch.zeros(1, 23, 2, device=device)

    best_response = -float('inf')
    best_theta = 0
    best_sf = 2.0
    optimal_grating = None

    n_combinations = len(thetas) * len(sfs)

    print(f"\n{'='*60}")
    print(f"[Phase 1] 2D Grid Search for Neuron {neuron_idx}")
    print(f"  Directions (θ): {thetas}")
    print(f"  Spatial Frequencies (sf): {sfs}")
    print(f"  Total combinations: {len(thetas)} × {len(sfs)} = {n_combinations}")
    print(f"  Fixed: tf={tf}, amp={amplitude}")
    print(f"{'='*60}")

    # 2D Grid Search
    for theta in thetas:
        for sf in sfs:
            # 生成纯净光栅
            grating = create_drifting_grating(
                direction=theta,
                sf=sf,
                tf=tf,
                amplitude=amplitude,
                device=device
            )

            # 记录响应
            with torch.no_grad():
                output = model(grating, behavior)  # (1, 23, N_neurons)
                response = output[0, :, neuron_idx].mean().item()

            print(f"  θ={theta:3.0f}°, sf={sf:.1f}: response = {response:+.4f}")

            if response > best_response:
                best_response = response
                best_theta = theta
                best_sf = sf
                optimal_grating = grating.clone()

    print(f"\n  ★ [Phase 1 Locked] Neuron {neuron_idx} Prefers:")
    print(f"      Theta = {best_theta:.0f}°")
    print(f"      SF = {best_sf}")
    print(f"      Response = {best_response:+.4f}")
    print(f"{'='*60}\n")

    return optimal_grating, best_theta, best_sf, best_response


# ============================================================
# Phase 2: 先验 MEI 优化
# ============================================================

def apply_l2_constraint(stimulus: torch.Tensor, max_norm: float = 10.0) -> torch.Tensor:
    """L2 范数约束 - 限制刺激的对比度"""
    with torch.no_grad():
        norm = stimulus.norm()
        if norm > max_norm:
            stimulus.mul_(max_norm / norm)
    return stimulus


def apply_gaussian_blur(stimulus: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    """空间平滑 - 过滤高频对抗性噪点"""
    with torch.no_grad():
        stim_np = stimulus.cpu().numpy()
        n_frames = stim_np.shape[2]
        for t in range(n_frames):
            stim_np[0, 0, t] = gaussian_filter(stim_np[0, 0, t], sigma=sigma)
        stimulus.data = torch.from_numpy(stim_np).to(stimulus.device)
    return stimulus


def compute_spatial_tv_loss(stimulus: torch.Tensor) -> torch.Tensor:
    """空间 Total Variation Loss"""
    x = stimulus[0, 0]
    dx = torch.abs(x[:, :, 1:] - x[:, :, :-1])
    dy = torch.abs(x[:, 1:, :] - x[:, :-1, :])
    return dx.mean() + dy.mean()


def compute_temporal_tv_loss(stimulus: torch.Tensor) -> torch.Tensor:
    """时间 TV Loss"""
    temporal_diff = torch.abs(stimulus[:, :, 1:, :, :] - stimulus[:, :, :-1, :, :])
    return temporal_diff.mean()


def create_gradient_mask(size: int = 80, sigma: float = 15.0, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """创建高斯梯度掩码"""
    y, x = torch.meshgrid(
        torch.linspace(-size//2, size//2, size),
        torch.linspace(-size//2, size//2, size),
        indexing='ij'
    )
    mask = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    mask = mask / mask.max()
    return mask.view(1, 1, 1, size, size).to(device)


def create_center_mask(size: int = 80, sigma: float = 20.0) -> np.ndarray:
    """创建高斯中心遮罩"""
    y, x = np.mgrid[-size//2:size//2, -size//2:size//2]
    mask = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return mask.astype(np.float32)


def apply_center_mask(stimulus: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
    """将中心遮罩应用到每一帧"""
    with torch.no_grad():
        stim_np = stimulus.cpu().numpy()
        n_frames = stim_np.shape[2]
        for t in range(n_frames):
            stim_np[0, 0, t] *= mask
        stimulus.data = torch.from_numpy(stim_np).to(stimulus.device)
    return stimulus


def generate_informed_mei(
    model: nn.Module,
    neuron_idx: int,
    seed_grating: torch.Tensor,
    n_iterations: int = 200,
    learning_rate: float = 0.003,
    max_norm: float = 15.0,
    blur_sigma: float = 1.0,
    blur_interval: int = 10,
    lambda_spatial_tv: float = 1.0,
    lambda_temporal_tv: float = 0.01,
    grad_kernel_size: int = 5,
    grad_sigma: float = 1.5,
    value_clip_min: float = -3.0,
    value_clip_max: float = 3.0,
    center_mask_sigma: float = 20.0,
    device: torch.device = torch.device('cpu'),
    verbose: bool = True
) -> torch.Tensor:
    """
    Phase 2: 基于先验种子的 MEI 优化

    关键特性:
        - 使用最优光栅作为初始化种子 (已处于全局最优点附近)
        - 低学习率 (0.003) 微调，防止砸碎光栅结构
        - 保留空间高斯模糊 + 梯度掩码
        - 禁用时间维度 avg_pool3d (保留光栅的时间结构)
    """
    model.eval()

    # 用种子初始化
    stimulus = seed_grating.clone()
    stimulus.requires_grad = True

    behavior = torch.zeros(1, 23, 2, device=device)

    center_mask = create_center_mask(size=80, sigma=center_mask_sigma)
    gradient_mask = create_gradient_mask(size=80, sigma=15.0, device=device)

    optimizer = torch.optim.Adam([stimulus], lr=learning_rate)

    print(f"\n{'='*60}")
    print(f"[Phase 2] Informed MEI Optimization for Neuron {neuron_idx}")
    print(f"  [Parametric-to-Pixel: Using grating seed]")
    print(f"  Iterations: {n_iterations}")
    print(f"  Learning rate: {learning_rate} (LOW - preserving grating structure)")
    print(f"  ** NO temporal avg_pool3d - preserving grating motion **")
    print(f"{'='*60}\n")

    for step in range(n_iterations):
        output = model(stimulus, behavior)
        target_response = output[0, :, neuron_idx]
        response_loss = -target_response.mean()

        spatial_tv = compute_spatial_tv_loss(stimulus)
        temporal_tv = compute_temporal_tv_loss(stimulus)

        loss = (response_loss +
                lambda_spatial_tv * spatial_tv +
                lambda_temporal_tv * temporal_tv)

        optimizer.zero_grad()
        loss.backward()

        # 梯度平滑 (仅空间，禁用时间池化！)
        smoothed_grad = torchvision.transforms.functional.gaussian_blur(
            stimulus.grad.data.view(33, 1, 80, 80),
            kernel_size=grad_kernel_size,
            sigma=grad_sigma
        )
        stimulus.grad.data = smoothed_grad.view(1, 1, 33, 80, 80)

        # 梯度掩码
        stimulus.grad.data = stimulus.grad.data * gradient_mask

        optimizer.step()

        # 后处理约束
        stimulus.data = torch.clamp(stimulus.data, value_clip_min, value_clip_max)
        stimulus = apply_l2_constraint(stimulus, max_norm=max_norm)

        if step % blur_interval == 0 and step > 0:
            stimulus = apply_gaussian_blur(stimulus, sigma=blur_sigma)

        if step % 20 == 0 and step > 0:
            stimulus = apply_center_mask(stimulus, center_mask)

        if verbose and (step % 50 == 0 or step == n_iterations - 1):
            with torch.no_grad():
                current_response = target_response.mean().item()
                current_norm = stimulus.norm().item()
            print(f"  Step {step:4d}/{n_iterations}: "
                  f"Response = {current_response:.4f}, "
                  f"Loss = {loss.item():.4f}, "
                  f"Norm = {current_norm:.2f}")

    print(f"\n[DONE] Informed MEI generation completed for Neuron {neuron_idx}")

    return stimulus.detach()


# ============================================================
# 可视化函数 (支持自定义标题)
# ============================================================

def visualize_stimulus_grid(
    stimulus_tensor: np.ndarray,
    save_path: Path,
    neuron_idx: int,
    title_prefix: str = "MEI",
    subtitle: str = None,
    cmap: str = 'gray'
):
    """
    将 33 帧刺激绘制成网格图像

    Args:
        stimulus_tensor: 刺激张量 (1, 1, 33, 80, 80) 或 (33, 80, 80)
        save_path: 保存路径
        neuron_idx: 神经元索引
        title_prefix: 标题前缀 (e.g., "Original Strongest", "Best Grating Seed", "Final MEI")
        subtitle: 可选副标题 (e.g., "Peak Response: 2.5")
        cmap: 颜色映射
    """
    if stimulus_tensor.ndim == 5:
        stimulus_tensor = stimulus_tensor[0, 0]

    n_frames = stimulus_tensor.shape[0]
    n_cols = 9
    n_rows = (n_frames + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten()

    for i in range(n_frames):
        ax = axes[i]
        frame = stimulus_tensor[i]
        frame_min = frame.min()
        frame_max = frame.max()
        frame_norm = (frame - frame_min) / (frame_max - frame_min + 1e-8)
        ax.imshow(frame_norm, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f't={i}', fontsize=10)
        ax.axis('off')

    for i in range(n_frames, len(axes)):
        axes[i].axis('off')

    title = f'{title_prefix} for Neuron {neuron_idx}'
    if subtitle:
        title += f' ({subtitle})'
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[Saved] Grid visualization: {save_path}")


def save_stimulus_movie(
    stimulus_tensor: np.ndarray,
    save_path: Path,
    neuron_idx: int,
    title_prefix: str = "MEI",
    subtitle: str = None,
    fps: int = 3,
    cmap: str = 'gray'
):
    """
    将 33 帧刺激保存为视频文件

    Args:
        stimulus_tensor: 刺激张量 (1, 1, 33, 80, 80) 或 (33, 80, 80)
        save_path: 保存路径 (.mp4 或 .gif)
        neuron_idx: 神经元索引
        title_prefix: 标题前缀
        subtitle: 可选副标题
        fps: 帧率
        cmap: 颜色映射
    """
    if stimulus_tensor.ndim == 5:
        stimulus_tensor = stimulus_tensor[0, 0]

    n_frames = stimulus_tensor.shape[0]
    min_val = stimulus_tensor.min()
    max_val = stimulus_tensor.max()
    stim_norm = (stimulus_tensor - min_val) / (max_val - min_val + 1e-8)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.tight_layout()

    im = ax.imshow(stim_norm[0], cmap=cmap, vmin=0, vmax=1, animated=True)
    title_text = f'{title_prefix} Neuron {neuron_idx}'
    if subtitle:
        title_text += f' ({subtitle})'
    title = ax.set_title(f'{title_text} - Frame 0/{n_frames-1}', fontsize=12)
    ax.axis('off')

    def update(frame_idx):
        im.set_array(stim_norm[frame_idx])
        title.set_text(f'{title_text} - Frame {frame_idx}/{n_frames-1}')
        return [im, title]

    anim = FuncAnimation(
        fig, update,
        frames=n_frames,
        interval=1000 // fps,
        blit=True,
        repeat=True
    )

    if save_path.suffix == '.gif':
        anim.save(save_path, writer=PillowWriter(fps=fps), dpi=100)
    else:
        try:
            anim.save(save_path, writer='ffmpeg', fps=fps, dpi=100)
        except Exception as e:
            print(f"[Warning] ffmpeg not available: {e}")
            gif_path = save_path.with_suffix('.gif')
            anim.save(gif_path, writer=PillowWriter(fps=fps), dpi=100)
            save_path = gif_path

    plt.close(fig)
    print(f"[Saved] Movie: {save_path} (fps={fps}, frames={n_frames})")


# ============================================================
# 模型加载函数
# ============================================================

def load_trained_model(
    checkpoint_path: Path,
    n_neurons: int,
    device: torch.device
) -> nn.Module:
    """加载训练好的模型"""
    model = create_model(
        n_neurons=n_neurons,
        model_type="DeepRetina3D",
        device=device
    )

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[Loaded] Model weights from: {checkpoint_path}")
        if 'val_r' in checkpoint:
            print(f"  Validation R: {checkpoint['val_r']:.4f}")
    else:
        print(f"[Warning] Checkpoint not found: {checkpoint_path}")
        print("  Using random weights (for testing only)")

    model.to(device)
    return model


# ============================================================
# 主函数 (三维对比组合管线)
# ============================================================

def main():
    """
    主入口函数 - 三维对比组合管线

    流程:
        1. Phase 0: 在真实数据集中搜索最强刺激
        2. Phase 1: 2D Grid Search 找最优光栅
        3. Phase 2: 基于先验种子的 MEI 优化
    """

    # ==========================================
    # 配置参数
    # ==========================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 模型参数
    N_NEURONS = 50

    # ============================================================
    # Phase 0: 数据路径配置
    # ------------------------------------------------------------
    # 设置为 None 可跳过 Phase 0 (只运行 Phase 1 + Phase 2)
    # ============================================================
    DATA_MAT_PATH = PROJECT_ROOT / "data" / "training.mat"  # TODO: 修改为实际路径
    PHASE0_BATCH_SIZE = 64

    # 创建 Phase 0 DataLoader
    val_dataloader = None
    if DATA_MAT_PATH.exists():
        print(f"\n[Phase 0] Creating Stimulus DataLoader...")
        val_dataloader = create_stimulus_dataloader(
            mat_path=str(DATA_MAT_PATH),
            batch_size=PHASE0_BATCH_SIZE,
            rf_diameter=80,
            normalize=True,
            expand_factor=30
        )
    else:
        print(f"\n[Phase 0] Data file not found: {DATA_MAT_PATH}")
        print(f"  Phase 0 will be skipped. Only Phase 1 + Phase 2 will run.")
        print(f"  To enable Phase 0, set DATA_MAT_PATH to your .mat file.")

    # Phase 1 参数 (2D Grid Search: 方向 × 空间频率)
    DIRECTIONS = [0, 45, 90, 135, 180, 225, 270, 315]
    SPATIAL_FREQUENCIES = [1.5, 2.5, 3.5]  # 粗、中、细三种条纹
    GRATING_TF = 2.0       # 时间频率
    GRATING_AMPLITUDE = 0.8

    # Phase 2 参数 (优化)
    TARGET_NEURONS = range(10,20)
    N_ITERATIONS = 100     # 较少步数 (已接近最优)
    LEARNING_RATE = 0.003   # 低学习率 (防止砸碎光栅)
    MAX_NORM = 15.0

    # 梯度平滑参数
    GRAD_KERNEL_SIZE = 5
    GRAD_SIGMA = 1.5

    # TV Loss 参数
    LAMBDA_SPATIAL_TV = 1.0
    LAMBDA_TEMPORAL_TV = 0.01  # 极弱，释放运动

    # 数值边界
    VALUE_CLIP_MIN = -3.0
    VALUE_CLIP_MAX = 3.0

    # 后处理
    BLUR_SIGMA = 1.0
    BLUR_INTERVAL = 10
    CENTER_MASK_SIGMA = 20.0

    # 路径配置
    CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pth"
    OUTPUT_DIR = PROJECT_ROOT / "three_phase_comparison"
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ==========================================
    # 加载模型
    # ==========================================
    model = load_trained_model(
        checkpoint_path=CHECKPOINT_PATH,
        n_neurons=N_NEURONS,
        device=device
    )

    # ==========================================
    # 批量生成三维对比组合
    # ==========================================
    VIDEO_FPS = 3

    for neuron_idx in TARGET_NEURONS:
        print(f"\n{'#'*70}")
        print(f"# Processing Neuron {neuron_idx} (Three-Phase Comparison)")
        print(f"{'#'*70}")

        # ==========================================
        # 创建神经元专属文件夹
        # ==========================================
        neuron_dir = OUTPUT_DIR / f"neuron_{neuron_idx}"
        neuron_dir.mkdir(exist_ok=True)

        # ==========================================
        # Phase 0: 真实最强刺激
        # ==========================================
        if val_dataloader is not None:
            original_stimulus, peak_response = find_strongest_natural_stimulus(
                model=model,
                dataloader=val_dataloader,
                neuron_idx=neuron_idx,
                device=device
            )

            # 保存 Original Strongest
            original_np = original_stimulus.cpu().numpy()

            visualize_stimulus_grid(
                original_np,
                neuron_dir / "original_strongest.png",
                neuron_idx=neuron_idx,
                title_prefix="Original Strongest",
                subtitle=f"Peak: {peak_response:.2f}"
            )

            save_stimulus_movie(
                original_np,
                neuron_dir / "original_strongest.mp4",
                neuron_idx=neuron_idx,
                title_prefix="Original Strongest",
                subtitle=f"Peak: {peak_response:.2f}",
                fps=VIDEO_FPS
            )
        else:
            print(f"\n[Skipping Phase 0] No DataLoader provided")

        # ==========================================
        # Phase 1: 最佳光栅种子
        # ==========================================
        optimal_grating, best_theta, best_sf, best_response = find_optimal_grating(
            model=model,
            neuron_idx=neuron_idx,
            thetas=DIRECTIONS,
            sfs=SPATIAL_FREQUENCIES,
            tf=GRATING_TF,
            amplitude=GRATING_AMPLITUDE,
            device=device
        )

        # 保存 Best Grating Seed
        grating_np = optimal_grating.cpu().numpy()

        visualize_stimulus_grid(
            grating_np,
            neuron_dir / "best_grating_seed.png",
            neuron_idx=neuron_idx,
            title_prefix="Best Grating Seed",
            subtitle=f"Dir: {best_theta:.0f}°, SF: {best_sf:.1f}"
        )

        save_stimulus_movie(
            grating_np,
            neuron_dir / "best_grating_seed.mp4",
            neuron_idx=neuron_idx,
            title_prefix="Best Grating Seed",
            subtitle=f"Dir: {best_theta:.0f}°, SF: {best_sf:.1f}",
            fps=VIDEO_FPS
        )

        # ==========================================
        # Phase 2: 最终 MEI
        # ==========================================
        mei_tensor = generate_informed_mei(
            model=model,
            neuron_idx=neuron_idx,
            seed_grating=optimal_grating,
            n_iterations=N_ITERATIONS,
            learning_rate=LEARNING_RATE,
            max_norm=MAX_NORM,
            blur_sigma=BLUR_SIGMA,
            blur_interval=BLUR_INTERVAL,
            lambda_spatial_tv=LAMBDA_SPATIAL_TV,
            lambda_temporal_tv=LAMBDA_TEMPORAL_TV,
            grad_kernel_size=GRAD_KERNEL_SIZE,
            grad_sigma=GRAD_SIGMA,
            value_clip_min=VALUE_CLIP_MIN,
            value_clip_max=VALUE_CLIP_MAX,
            center_mask_sigma=CENTER_MASK_SIGMA,
            device=device,
            verbose=True
        )

        # 保存 Final MEI
        mei_np = mei_tensor.cpu().numpy()

        np.save(neuron_dir / "final_mei.npy", mei_np)
        print(f"[Saved] MEI tensor: {neuron_dir / 'final_mei.npy'}")
        print(f"  Shape: {mei_np.shape}")

        visualize_stimulus_grid(
            mei_np,
            neuron_dir / "final_mei.png",
            neuron_idx=neuron_idx,
            title_prefix="Final Informed MEI",
            subtitle=f"Seed: θ={best_theta:.0f}°, SF={best_sf:.1f}"
        )

        save_stimulus_movie(
            mei_np,
            neuron_dir / "final_mei.mp4",
            neuron_idx=neuron_idx,
            title_prefix="Final Informed MEI",
            subtitle=f"Seed: θ={best_theta:.0f}°, SF={best_sf:.1f}",
            fps=VIDEO_FPS
        )

        # ==========================================
        # 神经元总结
        # ==========================================
        print(f"\n{'='*60}")
        print(f"Neuron {neuron_idx} Summary")
        print(f"{'='*60}")
        if val_dataloader is not None:
            print(f"  Original Peak Response: {peak_response:.4f}")
        print(f"  Best Grating: θ={best_theta:.0f}°, SF={best_sf:.1f}, Response={best_response:.4f}")
        print(f"  Final MEI shape: {mei_np.shape}")
        print(f"  Final MEI range: [{mei_np.min():.4f}, {mei_np.max():.4f}]")
        print(f"  Output directory: {neuron_dir}")
        print(f"{'='*60}")

    # ==========================================
    # 批量总结
    # ==========================================
    print(f"\n{'='*70}")
    print("Three-Phase Comparison Generation Complete")
    print(f"{'='*70}")
    print(f"  Processed neurons: {TARGET_NEURONS}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"\n  Directory structure:")
    print(f"  {OUTPUT_DIR}/")
    for neuron_idx in TARGET_NEURONS:
        print(f"  ├── neuron_{neuron_idx}/")
        print(f"  │   ├── original_strongest.png")
        print(f"  │   ├── original_strongest.mp4")
        print(f"  │   ├── best_grating_seed.png")
        print(f"  │   ├── best_grating_seed.mp4")
        print(f"  │   ├── final_mei.png")
        print(f"  │   └── final_mei.mp4")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
