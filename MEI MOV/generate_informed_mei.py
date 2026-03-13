#!C:/Users/vipuser/anaconda3/envs/chatgpt/python.exe
# -*- coding: utf-8 -*-
"""
MEI 生成脚本 - Parametric-to-Pixel (先验初始化) 管线

这套流程使用了 Parametric-to-Pixel 方法，用真实偏好方向打破局部最优解的迷彩诅咒。

两阶段流程:
    Phase 1: 虚拟电生理测向 - 找到神经元的偏好方向光栅
    Phase 2: 先验 MEI 优化 - 用最优光栅作为种子进行微调
    Phase 3: 输出保存 - .npy, .png 网格, .mp4 视频

模型规格:
    - Stimulus 输入: (B, 1, 33, 80, 80)
    - Behavior 输入: (B, 23, 2) - 固定为全0 (Neutral State)
    - 模型输出: (B, 23, N_neurons) - Firing Rate

Author: Claude
Date: 2025-03
"""

import sys
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import create_model


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

    遍历验证集，找出导致目标神经元产生最高峰值放电率的 33 帧视频片段。
    这为三维对比组合提供"真实基准"。

    Args:
        model: 训练好的 DeepRetina3D 模型
        dataloader: 验证集 DataLoader (需实现)
        neuron_idx: 目标神经元索引
        device: 计算设备

    Returns:
        strongest_stimulus: 最强自然刺激 (1, 1, 33, 80, 80)
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
            # TODO: 根据你的数据格式调整 key 名称
            stimulus = batch['stimulus'].to(device)  # (B, 1, 33, 80, 80)
            behavior = batch['behavior'].to(device)  # (B, 23, 2)

            output = model(stimulus, behavior)  # (B, 23, N_neurons)

            # 找峰值响应
            neuron_output = output[:, :, neuron_idx]  # (B, 23)
            batch_peak = neuron_output.max(dim=1).values  # (B,)

            max_in_batch = batch_peak.max()
            if max_in_batch > peak_response:
                peak_response = max_in_batch
                max_idx = batch_peak.argmax()
                best_stimulus = stimulus[max_idx:max_idx+1].clone()
                print(f"  Found new peak: {peak_response:.4f} (batch {batch_idx})")

    if best_stimulus is None:
        # 如果没找到，生成随机噪声作为占位符
        print(f"  [Warning] No data found, using random noise as placeholder")
        best_stimulus = torch.randn(1, 1, 33, 80, 80, device=device) * 0.1
        peak_response = 0.0

    print(f"\n  ★ [Phase 0 Found] Neuron {neuron_idx} Peak Response: {peak_response:.4f}")
    print(f"{'='*60}\n")

    return best_stimulus, peak_response.item()


# ============================================================
# Phase 1: 虚拟电生理测向 (方向偏好测试)
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
# 正则化函数 (复用自 generate_mei.py)
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
    """空间 Total Variation Loss - 惩罚相邻像素差异"""
    x = stimulus[0, 0]  # (33, 80, 80)
    dx = torch.abs(x[:, :, 1:] - x[:, :, :-1])
    dy = torch.abs(x[:, 1:, :] - x[:, :-1, :])
    tv_loss = dx.mean() + dy.mean()
    return tv_loss


def compute_temporal_tv_loss(stimulus: torch.Tensor) -> torch.Tensor:
    """时间 TV Loss - 惩罚相邻帧之间的突变"""
    temporal_diff = torch.abs(stimulus[:, :, 1:, :, :] - stimulus[:, :, :-1, :, :])
    return temporal_diff.mean()


def create_gradient_mask(size: int = 80, sigma: float = 15.0, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    创建高斯梯度掩码，只允许中心区域更新

    物理意义: 迫使模型只能修改画面中心的像素，
    直接切断所有边缘的"迷彩"噪音生成。
    """
    y, x = torch.meshgrid(
        torch.linspace(-size//2, size//2, size),
        torch.linspace(-size//2, size//2, size),
        indexing='ij'
    )
    mask = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    mask = mask / mask.max()  # 归一化到 [0, 1]
    return mask.view(1, 1, 1, size, size).to(device)


def create_center_mask(size: int = 80, sigma: float = 20.0) -> np.ndarray:
    """创建高斯中心遮罩，抑制边缘噪声"""
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


# ============================================================
# Phase 2: 先验 MEI 优化
# ============================================================

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
    第二阶段：基于先验种子的 MEI 优化

    关键特性:
        - 使用最优光栅作为初始化种子 (已处于全局最优点附近)
        - 低学习率 (0.01) 微调，防止砸碎光栅结构
        - 保留空间高斯模糊 + 梯度掩码
        - 禁用时间维度 avg_pool3d (保留光栅的时间结构)

    Args:
        model: 训练好的 DeepRetina3D 模型
        neuron_idx: 目标神经元索引
        seed_grating: 来自 Phase 1 的最优光栅种子
        n_iterations: 优化迭代次数 (默认 200，较少因为已接近最优)
        learning_rate: Adam 学习率 (默认 0.01，低学习率防止砸碎光栅)
        max_norm: L2 范数约束阈值
        blur_sigma: 高斯模糊的 sigma (后处理)
        blur_interval: 每隔多少步应用一次后处理模糊
        lambda_spatial_tv: 空间 TV Loss 权重
        lambda_temporal_tv: 时间 TV Loss 权重 (极弱，释放运动)
        grad_kernel_size: 梯度平滑核大小
        grad_sigma: 梯度平滑 sigma
        value_clip_min: 像素值下界
        value_clip_max: 像素值上界
        center_mask_sigma: 中心遮罩高斯窗口的 sigma
        device: 计算设备
        verbose: 是否打印进度

    Returns:
        生成的 MEI 张量 (1, 1, 33, 80, 80)
    """
    model.eval()

    # ==========================================
    # 用种子初始化 stimulus (关键！)
    # ==========================================
    stimulus = seed_grating.clone()
    stimulus.requires_grad = True

    # Behavior 固定为全0 (Neutral State)
    behavior = torch.zeros(1, 23, 2, device=device)

    # ==========================================
    # 创建遮罩
    # ==========================================
    center_mask = create_center_mask(size=80, sigma=center_mask_sigma)
    gradient_mask = create_gradient_mask(size=80, sigma=15.0, device=device)

    # ==========================================
    # 设置优化器 (低学习率！)
    # ==========================================
    optimizer = torch.optim.Adam([stimulus], lr=learning_rate)

    # ==========================================
    # 优化循环
    # ==========================================
    print(f"\n{'='*60}")
    print(f"[Phase 2] Informed MEI Optimization for Neuron {neuron_idx}")
    print(f"  [Parametric-to-Pixel: Using grating seed]")
    print(f"  Iterations: {n_iterations}")
    print(f"  Learning rate: {learning_rate} (LOW - preserving grating structure)")
    print(f"  Max L2 norm: {max_norm}")
    print(f"  Gradient smoothing: kernel={grad_kernel_size}, sigma={grad_sigma}")
    print(f"  Spatial TV weight: {lambda_spatial_tv}")
    print(f"  Temporal TV weight: {lambda_temporal_tv} (WEAK - releasing motion)")
    print(f"  Value clipping: [{value_clip_min}, {value_clip_max}]")
    print(f"  Center mask sigma: {center_mask_sigma}")
    print(f"  ** NO temporal avg_pool3d - preserving grating motion **")
    print(f"{'='*60}\n")

    for step in range(n_iterations):
        # 前向传播
        output = model(stimulus, behavior)  # (1, 23, N_neurons)

        # 计算响应损失: 负的平均响应 (因为 Adam 是最小化)
        target_response = output[0, :, neuron_idx]  # (23,)
        response_loss = -target_response.mean()

        # ==========================================
        # 时空双重 TV Loss
        # ==========================================
        spatial_tv = compute_spatial_tv_loss(stimulus)
        temporal_tv = compute_temporal_tv_loss(stimulus)

        # 总损失
        loss = (response_loss +
                lambda_spatial_tv * spatial_tv +
                lambda_temporal_tv * temporal_tv)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # ==========================================
        # 梯度平滑 (仅空间，禁用时间池化！)
        # ==========================================
        # 1. 空间平滑 (Gaussian Blur)
        # 这从根本上阻止了全连接层产生高频对抗样本
        smoothed_grad = torchvision.transforms.functional.gaussian_blur(
            stimulus.grad.data.view(33, 1, 80, 80),
            kernel_size=grad_kernel_size,
            sigma=grad_sigma
        )
        stimulus.grad.data = smoothed_grad.view(1, 1, 33, 80, 80)

        # 2. 梯度掩码 (切断边缘噪音生成)
        stimulus.grad.data = stimulus.grad.data * gradient_mask

        # 注意: 禁用时间维度 avg_pool3d！
        # 保留光栅的时间结构，不破坏运动特征

        optimizer.step()

        # ==========================================
        # 后处理约束
        # ==========================================
        # 1. 数值截断
        stimulus.data = torch.clamp(stimulus.data, value_clip_min, value_clip_max)

        # 2. L2 范数约束
        stimulus = apply_l2_constraint(stimulus, max_norm=max_norm)

        # 3. 空间平滑 (每隔 blur_interval 步)
        if step % blur_interval == 0 and step > 0:
            stimulus = apply_gaussian_blur(stimulus, sigma=blur_sigma)

        # 4. 中心遮罩 (每隔 20 步)
        if step % 20 == 0 and step > 0:
            stimulus = apply_center_mask(stimulus, center_mask)

        # 打印进度
        if verbose and (step % 50 == 0 or step == n_iterations - 1):
            with torch.no_grad():
                current_response = target_response.mean().item()
                current_norm = stimulus.norm().item()
                current_spatial_tv = spatial_tv.item()
                current_temporal_tv = temporal_tv.item()
            print(f"  Step {step:4d}/{n_iterations}: "
                  f"Response = {current_response:.4f}, "
                  f"Loss = {loss.item():.4f}, "
                  f"SpatialTV = {current_spatial_tv:.4f}, "
                  f"TemporalTV = {current_temporal_tv:.4f}, "
                  f"Norm = {current_norm:.2f}")

    print(f"\n[DONE] Informed MEI generation completed for Neuron {neuron_idx}")

    return stimulus.detach()


# ============================================================
# 可视化函数
# ============================================================

def visualize_mei_grid(
    mei_tensor: np.ndarray,
    save_path: Path,
    neuron_idx: int,
    best_direction: float = None,
    cmap: str = 'gray'
):
    """将 33 帧 MEI 绘制成网格图像"""
    if mei_tensor.ndim == 5:
        mei_tensor = mei_tensor[0, 0]

    n_frames = mei_tensor.shape[0]
    n_cols = 9
    n_rows = (n_frames + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten()

    for i in range(n_frames):
        ax = axes[i]
        frame = mei_tensor[i]
        frame_min = frame.min()
        frame_max = frame.max()
        frame_norm = (frame - frame_min) / (frame_max - frame_min + 1e-8)
        ax.imshow(frame_norm, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f't={i}', fontsize=10)
        ax.axis('off')

    for i in range(n_frames, len(axes)):
        axes[i].axis('off')

    title = f'Informed MEI for Neuron {neuron_idx}'
    if best_direction is not None:
        title += f' (Pref. Dir: {best_direction:.0f}°)'
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[Saved] MEI grid visualization: {save_path}")


def save_mei_movie(
    mei_tensor: np.ndarray,
    save_path: Path,
    neuron_idx: int,
    best_direction: float = None,
    fps: int = 3,
    cmap: str = 'gray'
):
    """将 33 帧 MEI 保存为视频文件"""
    if mei_tensor.ndim == 5:
        mei_tensor = mei_tensor[0, 0]

    n_frames = mei_tensor.shape[0]
    min_val = mei_tensor.min()
    max_val = mei_tensor.max()
    mei_norm = (mei_tensor - min_val) / (max_val - min_val + 1e-8)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.tight_layout()

    im = ax.imshow(mei_norm[0], cmap=cmap, vmin=0, vmax=1, animated=True)
    title_text = f'Informed MEI Neuron {neuron_idx}'
    if best_direction is not None:
        title_text += f' (Dir: {best_direction:.0f}°)'
    title = ax.set_title(f'{title_text} - Frame 0/{n_frames-1}', fontsize=12)
    ax.axis('off')

    def update(frame_idx):
        im.set_array(mei_norm[frame_idx])
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
    print(f"[Saved] MEI movie: {save_path} (fps={fps}, frames={n_frames})")


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
# Phase 3: 主循环
# ============================================================

def main():
    """
    主入口函数 - Parametric-to-Pixel MEI 生成管线

    流程:
        1. Phase 1: 虚拟电生理测向 (找最优光栅)
        2. Phase 2: 先验 MEI 优化 (微调)
        3. Phase 3: 保存输出
    """

    # ==========================================
    # 配置参数
    # ==========================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 模型参数
    N_NEURONS = 50

    # Phase 1 参数 (2D Grid Search: 方向 × 空间频率)
    DIRECTIONS = [0, 45, 90, 135, 180, 225, 270, 315]
    SPATIAL_FREQUENCIES = [1.5, 2.5, 3.5]  # 粗、中、细三种条纹
    GRATING_TF = 2.0       # 时间频率
    GRATING_AMPLITUDE = 0.8

    # Phase 2 参数 (优化)
    TARGET_NEURONS = [0, 1, 2, 3, 4]
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
    OUTPUT_DIR = PROJECT_ROOT / "informed_mei_results"
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
    # 批量生成 Informed MEI
    # ==========================================
    VIDEO_FPS = 3

    for neuron_idx in TARGET_NEURONS:
        print(f"\n{'#'*70}")
        print(f"# Processing Neuron {neuron_idx} (Parametric-to-Pixel Pipeline)")
        print(f"{'#'*70}")

        # ==========================================
        # Phase 1: 虚拟电生理测向
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
        best_direction = best_theta
        # ==========================================
        # Phase 2: 先验 MEI 优化
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

        # ==========================================
        # Phase 3: 保存结果
        # ==========================================
        mei_np = mei_tensor.cpu().numpy()

        # 保存 .npy
        npy_path = OUTPUT_DIR / f"informed_mei_neuron_{neuron_idx}.npy"
        np.save(npy_path, mei_np)
        print(f"[Saved] MEI tensor: {npy_path}")
        print(f"  Shape: {mei_np.shape}")

        # 保存可视化
        png_path = OUTPUT_DIR / f"informed_mei_neuron_{neuron_idx}.png"
        visualize_mei_grid(mei_np, png_path, neuron_idx, best_direction)

        # 保存视频
        mp4_path = OUTPUT_DIR / f"informed_mei_neuron_{neuron_idx}.mp4"
        gif_path = OUTPUT_DIR / f"informed_mei_neuron_{neuron_idx}.gif"

        try:
            save_mei_movie(mei_np, mp4_path, neuron_idx, best_direction, fps=VIDEO_FPS)
        except Exception as e:
            print(f"[Warning] MP4 save failed: {e}")
            try:
                save_mei_movie(mei_np, gif_path, neuron_idx, best_direction, fps=VIDEO_FPS)
            except Exception as e2:
                print(f"[Error] Video generation failed: {e2}")

        # 单神经元总结
        print(f"\n{'='*60}")
        print(f"Neuron {neuron_idx} Summary")
        print(f"{'='*60}")
        print(f"  Preferred direction: {best_direction:.0f}°")
        print(f"  Seed response: {best_response:.4f}")
        print(f"  MEI shape: {mei_np.shape}")
        print(f"  MEI value range: [{mei_np.min():.4f}, {mei_np.max():.4f}]")
        print(f"  MEI L2 norm: {np.linalg.norm(mei_np):.4f}")
        print(f"{'='*60}")

    # ==========================================
    # 批量总结
    # ==========================================
    print(f"\n{'='*70}")
    print("Parametric-to-Pixel MEI Generation Complete")
    print(f"{'='*70}")
    print(f"  Processed neurons: {TARGET_NEURONS}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
