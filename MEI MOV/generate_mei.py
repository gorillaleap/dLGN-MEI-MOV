#!C:/Users/vipuser/anaconda3/envs/chatgpt/python.exe
# -*- coding: utf-8 -*-
"""
MEI (Maximum Excitatory Input) 生成脚本

使用梯度上升优化输入刺激，以最大化特定神经元的响应。

模型规格:
    - Stimulus 输入: (B, 1, 33, 80, 80)
    - Behavior 输入: (B, 23, 2) - 固定为全0 (Neutral State)
    - 模型输出: (B, 23, N_neurons) - Firing Rate

Author: Claude
Date: 2025-03
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter
import torchvision

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import create_model


# ============================================================
# 正则化函数
# ============================================================

def apply_l2_constraint(stimulus: torch.Tensor, max_norm: float = 10.0) -> torch.Tensor:
    """
    L2 范数约束 - 限制刺激的对比度

    如果 stimulus 的 L2 范数超过 max_norm，则等比例缩放回来。
    这防止了优化过程中对比度无限放大。

    Args:
        stimulus: 输入刺激张量 (1, 1, 33, 80, 80)
        max_norm: 最大允许的 L2 范数

    Returns:
        约束后的 stimulus (原地修改)
    """
    with torch.no_grad():
        norm = stimulus.norm()
        if norm > max_norm:
            stimulus.mul_(max_norm / norm)
            # print(f"  [L2 Constraint] norm={norm:.2f} -> scaled to {max_norm}")
    return stimulus


def apply_gaussian_blur(stimulus: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    """
    空间平滑 - 过滤高频对抗性噪点

    对每一帧独立应用高斯模糊。

    Args:
        stimulus: 输入刺激张量 (1, 1, 33, 80, 80)
        sigma: 高斯核的标准差

    Returns:
        平滑后的 stimulus
    """
    with torch.no_grad():
        stim_np = stimulus.cpu().numpy()
        n_frames = stim_np.shape[2]

        for t in range(n_frames):
            # 对每一帧应用高斯模糊
            stim_np[0, 0, t] = gaussian_filter(stim_np[0, 0, t], sigma=sigma)

        stimulus.data = torch.from_numpy(stim_np).to(stimulus.device)

    return stimulus


def compute_spatial_tv_loss(stimulus: torch.Tensor) -> torch.Tensor:
    """
    空间 Total Variation Loss - 惩罚相邻像素差异

    对每一帧的 x, y 方向计算梯度差的 L1 范数。
    逼迫生成平滑的视觉特征，消除椒盐噪声。

    Args:
        stimulus: 输入刺激张量 (1, 1, 33, 80, 80)

    Returns:
        tv_loss: 标量损失值
    """
    # 提取空间维度
    x = stimulus[0, 0]  # (33, 80, 80)

    # x 方向梯度 (水平方向)
    dx = torch.abs(x[:, :, 1:] - x[:, :, :-1])  # (33, 80, 79)
    # y 方向梯度 (垂直方向)
    dy = torch.abs(x[:, 1:, :] - x[:, :-1, :])  # (33, 79, 80)

    # TV loss = 梯度差的 L1 范数
    tv_loss = dx.mean() + dy.mean()

    return tv_loss


def compute_temporal_tv_loss(stimulus: torch.Tensor) -> torch.Tensor:
    """
    时间 TV Loss - 惩罚相邻帧之间的突变

    确保时间维度的平滑性，避免帧间闪烁。

    Args:
        stimulus: 输入刺激张量 (1, 1, 33, 80, 80)

    Returns:
        temporal_tv_loss: 标量损失值
    """
    # 计算相邻帧的差异
    temporal_diff = torch.abs(stimulus[:, :, 1:, :, :] - stimulus[:, :, :-1, :, :])

    return temporal_diff.mean()


def create_center_mask(size: int = 80, sigma: float = 20.0) -> np.ndarray:
    """
    创建高斯中心遮罩，抑制边缘噪声

    Args:
        size: 空间尺寸 (80)
        sigma: 高斯窗口标准差 (控制感受野大小)

    Returns:
        mask: (80, 80) 中心为 1，边缘趋近 0
    """
    y, x = np.mgrid[-size//2:size//2, -size//2:size//2]
    mask = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return mask.astype(np.float32)


def create_gradient_mask(size: int = 80, sigma: float = 15.0, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    创建高斯���度掩码，只允许中心区域更新

    物理意义: 迫使模型只能修改画面中心的像素，
    直接切断所有边缘的"迷彩"噪音生成。

    Args:
        size: 空间尺寸 (80)
        sigma: 高斯窗口标准差 (控制感受野大小)
        device: 计算设备

    Returns:
        mask: (1, 1, 1, 80, 80) 归一化到 [0, 1]
    """
    y, x = torch.meshgrid(
        torch.linspace(-size//2, size//2, size),
        torch.linspace(-size//2, size//2, size),
        indexing='ij'
    )
    mask = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    mask = mask / mask.max()  # 归一化到 [0, 1]
    return mask.view(1, 1, 1, size, size).to(device)


def apply_center_mask(stimulus: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
    """
    将中心遮罩应用到每一帧

    Args:
        stimulus: 输入刺激张量 (1, 1, 33, 80, 80)
        mask: 中心遮罩 (80, 80)

    Returns:
        应用遮罩后的 stimulus (原地修改)
    """
    with torch.no_grad():
        stim_np = stimulus.cpu().numpy()
        n_frames = stim_np.shape[2]

        for t in range(n_frames):
            stim_np[0, 0, t] *= mask

        stimulus.data = torch.from_numpy(stim_np).to(stimulus.device)

    return stimulus


# ============================================================
# 可视化函数
# ============================================================

def visualize_mei_grid(
    mei_tensor: np.ndarray,
    save_path: Path,
    neuron_idx: int,
    cmap: str = 'gray'
):
    """
    将 33 帧 MEI 绘制成网格图像

    Args:
        mei_tensor: MEI 张量 (1, 1, 33, 80, 80) 或 (33, 80, 80)
        save_path: 保存路径
        neuron_idx: 神经元索引 (用于标题)
        cmap: 颜色映射
    """
    # 处理输入形状
    if mei_tensor.ndim == 5:
        mei_tensor = mei_tensor[0, 0]  # (33, 80, 80)

    n_frames = mei_tensor.shape[0]
    n_cols = 9
    n_rows = (n_frames + n_cols - 1) // n_cols  # 向上取整

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten()

    for i in range(n_frames):
        ax = axes[i]
        frame = mei_tensor[i]

        # 归一化到 [0, 1] 以便显示
        frame_min = frame.min()
        frame_max = frame.max()
        frame_norm = (frame - frame_min) / (frame_max - frame_min + 1e-8)

        ax.imshow(frame_norm, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f't={i}', fontsize=10)
        ax.axis('off')

    # 隐藏多余的子图
    for i in range(n_frames, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'MEI for Neuron {neuron_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[Saved] MEI grid visualization: {save_path}")


# ============================================================
# 视频生成函数
# ============================================================

def normalize_frames_for_video(mei_tensor: np.ndarray) -> np.ndarray:
    """
    将 MEI 帧归一化到 [0, 255] uint8

    用于视频编码器，确保像素值在有效范围内。

    Args:
        mei_tensor: MEI 张量 (33, 80, 80) 或 (1, 1, 33, 80, 80)

    Returns:
        归一化后的帧 (33, 80, 80) uint8
    """
    # 处理输入形状
    if mei_tensor.ndim == 5:
        mei_tensor = mei_tensor[0, 0]  # (33, 80, 80)

    # 全局归一化 (保持帧间对比度一致)
    min_val = mei_tensor.min()
    max_val = mei_tensor.max()
    mei_norm = (mei_tensor - min_val) / (max_val - min_val + 1e-8)

    # 转换为 uint8
    mei_uint8 = (mei_norm * 255).astype(np.uint8)

    return mei_uint8


def save_mei_movie(
    mei_tensor: np.ndarray,
    save_path: Path,
    neuron_idx: int,
    fps: int = 3,
    cmap: str = 'gray'
):
    """
    将 33 帧 MEI 保存为视频文件 (.mp4 或 .gif)

    使用 matplotlib.animation 生成动画，便于观察感受野的时间演变。

    Args:
        mei_tensor: MEI 张量 (1, 1, 33, 80, 80) 或 (33, 80, 80)
        save_path: 保存路径 (.mp4 或 .gif)
        neuron_idx: 神经元索引 (用于标题)
        fps: 帧率 (默认 3，33帧约11秒)
        cmap: 颜色映射
    """
    # 处理输入形状
    if mei_tensor.ndim == 5:
        mei_tensor = mei_tensor[0, 0]  # (33, 80, 80)

    n_frames = mei_tensor.shape[0]

    # 全局归一化 (保持帧间对比度一致)
    min_val = mei_tensor.min()
    max_val = mei_tensor.max()
    mei_norm = (mei_tensor - min_val) / (max_val - min_val + 1e-8)

    # 创建图形
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.tight_layout()

    # 初始化图像
    im = ax.imshow(mei_norm[0], cmap=cmap, vmin=0, vmax=1, animated=True)
    title = ax.set_title(f'MEI Neuron {neuron_idx} - Frame 0/{n_frames-1}', fontsize=12)
    ax.axis('off')

    def update(frame_idx):
        """更新每一帧"""
        im.set_array(mei_norm[frame_idx])
        title.set_text(f'MEI Neuron {neuron_idx} - Frame {frame_idx}/{n_frames-1}')
        return [im, title]

    # 创建动画
    anim = FuncAnimation(
        fig, update,
        frames=n_frames,
        interval=1000 // fps,  # 毫秒
        blit=True,
        repeat=True
    )

    # 保存视频
    if save_path.suffix == '.gif':
        # GIF 格式 (跨平台兼容性最好)
        anim.save(save_path, writer=PillowWriter(fps=fps), dpi=100)
    else:
        # MP4 格式 (需要 ffmpeg)
        try:
            anim.save(save_path, writer='ffmpeg', fps=fps, dpi=100)
        except Exception as e:
            # 如果 ffmpeg 不可用，自动切换到 GIF
            print(f"[Warning] ffmpeg not available: {e}")
            gif_path = save_path.with_suffix('.gif')
            anim.save(gif_path, writer=PillowWriter(fps=fps), dpi=100)
            save_path = gif_path

    plt.close(fig)
    print(f"[Saved] MEI movie: {save_path} (fps={fps}, frames={n_frames})")

def create_drifting_grating_seed(frames=33, height=80, width=80, device='cuda'):
    """生成一个带有随机方向和频率的微弱滑动光栅，作为初始种子"""
    # 1. 创建空间和时间网格
    y, x = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width), indexing='ij')
    t = torch.linspace(0, 1, frames)
    
    # 2. 随机生成方向(theta)、空间频率(sf)和时间频率(tf)
    theta = np.random.uniform(0, 2 * np.pi)
    sf = np.random.uniform(1.5, 3.5) # 控制条纹粗细
    tf = np.random.uniform(1.0, 3.0) # 控制滑动快慢
    
    # 3. 计算 3D 相位
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    x_theta = x_theta.unsqueeze(0).to(device) # 形状: (1, H, W)
    t_tensor = t.view(-1, 1, 1).to(device)    # 形状: (T, 1, 1)
    
    # 4. 生成正弦波并叠加微弱噪音
    phase = sf * 2 * np.pi * x_theta - tf * 2 * np.pi * t_tensor
    grating = torch.sin(phase)
    noise = torch.randn_like(grating) * 0.1
    
    # 5. 组合并调整形状为 (1, 1, 33, 80, 80)
    seed = (grating * 0.3 + noise) # 0.3 的振幅，让它很微弱
    return seed.view(1, 1, frames, height, width)

# ============================================================
# MEI 生成主函数
# ============================================================

def generate_mei(
    model: torch.nn.Module,
    neuron_idx: int = 0,
    n_iterations: int = 500,
    learning_rate: float = 0.1,
    max_norm: float = 15.0,
    blur_sigma: float = 1.5,
    blur_interval: int = 10,
    lambda_spatial_tv: float = 1.0,
    lambda_temporal_tv: float = 0.5,
    grad_kernel_size: int = 5,
    grad_sigma: float = 1.5,
    value_clip_min: float = -3.0,
    value_clip_max: float = 3.0,
    center_mask_sigma: float = 20.0,
    device: torch.device = torch.device('cpu'),
    verbose: bool = True
) -> torch.Tensor:
    """
    使用梯度上升生成 MEI (终极版: 梯度平滑 + 时空双重正则化)

    核心创新: 在 loss.backward() 之后、optimizer.step() 之前，
    对梯度应用高斯模糊，从根本上切断全连接层产生高频对抗样本的物理途径。

    Args:
        model: 训练好的 DeepRetina3D 模型
        neuron_idx: 目标神经元索引
        n_iterations: 优化迭代次数
        learning_rate: Adam 学习率
        max_norm: L2 范数约束阈值
        blur_sigma: 高斯模糊的 sigma (后处理)
        blur_interval: 每隔多少步应用一次后处理模糊
        lambda_spatial_tv: 空间 TV Loss 权重 (惩罚空间高频)
        lambda_temporal_tv: 时间 TV Loss 权重 (惩罚帧间突变)
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
    model.eval()  # 设置为评估模式

    # ==========================================
    # 初始化 stimulus (低频初始化)
    # ==========================================
# 🚀 替换为这段光栅初始化（前提是你已经把 create_drifting_grating_seed 函数粘贴到了文件里）
    stimulus = create_drifting_grating_seed(frames=33, height=80, width=80, device=device)
    stimulus.requires_grad = True

    # Behavior 固定为全0 (Neutral State)
    behavior = torch.zeros(1, 23, 2, device=device)

    # ==========================================
    # 创建遮罩
    # ==========================================
    # 中心遮罩 (用于后处理，抑制边缘噪声)
    center_mask = create_center_mask(size=80, sigma=center_mask_sigma)

    # 梯度掩码 (用于梯度更新，强制只在中心更新)
    # 物理意义: 迫使模型只能修改画面中心的像素，切断边缘"迷彩"噪音
    gradient_mask = create_gradient_mask(size=80, sigma=15.0, device=device)

    # ==========================================
    # 设置优化器
    # ==========================================
    optimizer = torch.optim.Adam([stimulus], lr=learning_rate)

    # ==========================================
    # 优化循环
    # ==========================================
    print(f"\n{'='*60}")
    print(f"Generating MEI for Neuron {neuron_idx}")
    print(f"  [Anti-Adversarial Mode: Gradient Smoothing + Dual TV Loss]")
    print(f"  Iterations: {n_iterations}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Max L2 norm: {max_norm}")
    print(f"  Gradient smoothing: kernel={grad_kernel_size}, sigma={grad_sigma}")
    print(f"  Spatial TV weight: {lambda_spatial_tv}")
    print(f"  Temporal TV weight: {lambda_temporal_tv}")
    print(f"  Value clipping: [{value_clip_min}, {value_clip_max}]")
    print(f"  Center mask sigma: {center_mask_sigma}")
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
        # 梯度平滑 (空间 + 时间)
        # ==========================================
        # 对梯度应用高斯模糊，迫使优化器只能沿着平滑的方向更新
        # 这从根本上阻止了全连接层产生高频对抗样本
        # 1. 先进行空间平滑 (Gaussian Blur)
        # 保持 (1, 1, 33, 80, 80) 不变，但在 H 和 W 上模糊
        smoothed_grad = torchvision.transforms.functional.gaussian_blur(
            stimulus.grad.data.view(33, 1, 80, 80), 
            kernel_size=5, 
            sigma=1.5
        )
        stimulus.grad.data = smoothed_grad.view(1, 1, 33, 80, 80)

        # 3. 梯度掩码 (关键！切断边缘噪音生成)
        # 物理意义: 迫使模型只能修改画面中心的像素，切断边缘"迷彩"噪音
        stimulus.grad.data = stimulus.grad.data * gradient_mask
        # ==========================================

        optimizer.step()

        # ==========================================
        # 后处理约束
        # ==========================================
        # 1. 数值截断 (严格的生理学边界)
        stimulus.data = torch.clamp(stimulus.data, value_clip_min, value_clip_max)

        # 2. L2 范数约束
        stimulus = apply_l2_constraint(stimulus, max_norm=max_norm)

        # 3. 空间平滑 (每隔 blur_interval 步)
        if step % blur_interval == 0 and step > 0:
            stimulus = apply_gaussian_blur(stimulus, sigma=blur_sigma)

        # 4. 中心遮罩 (每隔 20 步，抑制边缘噪声)
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

    print(f"\n[DONE] MEI generation completed for Neuron {neuron_idx}")

    return stimulus.detach()


# ============================================================
# 模型加载函数
# ============================================================

def load_trained_model(
    checkpoint_path: Path,
    n_neurons: int,
    device: torch.device
) -> torch.nn.Module:
    """
    加载训练好的模型

    Args:
        checkpoint_path: 权重文件路径 (.pth)
        n_neurons: 神经元数量
        device: 计算设备

    Returns:
        加载权重后的模型
    """
    # 创建模型
    model = create_model(
        n_neurons=n_neurons,
        model_type="DeepRetina3D",
        device=device
    )

    # 加载权重
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
# 主函数
# ============================================================

def main():
    """主入口函数"""

    # ==========================================
    # 配置参数
    # ==========================================
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 模型参数
    N_NEURONS = 50  # 根据你的数据集调整

    # MEI 生成参数 (终极版: 梯度平滑 + 时空双重正则化 + 释放运动)
    TARGET_NEURONS = list(range(5))  # 批量筛选 0-4 号神经元
    N_ITERATIONS = 500   # 优化迭代次数
    LEARNING_RATE = 0.1  # 学习率
    MAX_NORM = 15.0      # L2 范数约束

    # 梯度平滑参数 (关键！切断高频对抗路径)
    GRAD_KERNEL_SIZE = 5   # 梯度平滑核大小
    GRAD_SIGMA = 1.5       # 梯度平滑 sigma

    # 时空双重 TV Loss
    LAMBDA_SPATIAL_TV = 1.0   # 空间 TV Loss 权重 (10x 增强)
    LAMBDA_TEMPORAL_TV = 0.01 # 时间 TV Loss 权重 (极弱，释放运动)

    # 数值边界
    VALUE_CLIP_MIN = -3.0  # 像素值下界
    VALUE_CLIP_MAX = 3.0   # 像素值上界

    # 后处理
    BLUR_SIGMA = 1.0       # 高斯模糊 sigma (掩码已控制噪音，允许中心稍锐利)
    BLUR_INTERVAL = 10     # 模糊间隔
    CENTER_MASK_SIGMA = 20.0  # 中心遮罩 sigma

    # 路径配置
    # TODO: 修改为你的权重文件路径
    CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pth"
    OUTPUT_DIR = PROJECT_ROOT / "mei_results"
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
    # 批量生成 MEI (筛选运动选择性神经元)
    # ==========================================
    VIDEO_FPS = 3  # 帧率: 每秒3帧, 33帧约11秒

    for neuron_idx in TARGET_NEURONS:
        print(f"\n{'#'*60}")
        print(f"# Processing Neuron {neuron_idx}")
        print(f"{'#'*60}")

        # 生成 MEI
        mei_tensor = generate_mei(
            model=model,
            neuron_idx=neuron_idx,
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
        # 保存结果
        # ==========================================
        # 转换为 NumPy
        mei_np = mei_tensor.cpu().numpy()

        # 保存 .npy 文件
        npy_path = OUTPUT_DIR / f"mei_neuron_{neuron_idx}.npy"
        np.save(npy_path, mei_np)
        print(f"[Saved] MEI tensor: {npy_path}")
        print(f"  Shape: {mei_np.shape}")

        # 保存可视化
        png_path = OUTPUT_DIR / f"mei_neuron_{neuron_idx}.png"
        visualize_mei_grid(mei_np, png_path, neuron_idx)

        # 保存视频
        mp4_path = OUTPUT_DIR / f"mei_neuron_{neuron_idx}.mp4"
        gif_path = OUTPUT_DIR / f"mei_neuron_{neuron_idx}.gif"

        try:
            save_mei_movie(mei_np, mp4_path, neuron_idx, fps=VIDEO_FPS)
        except Exception as e:
            print(f"[Warning] MP4 save failed: {e}")
            print("  Falling back to GIF format...")
            try:
                save_mei_movie(mei_np, gif_path, neuron_idx, fps=VIDEO_FPS)
            except Exception as e2:
                print(f"[Error] Video generation failed: {e2}")

        # 单神经元总结
        print(f"\n{'='*60}")
        print(f"Neuron {neuron_idx} Summary")
        print(f"{'='*60}")
        print(f"  MEI shape: {mei_np.shape}")
        print(f"  MEI value range: [{mei_np.min():.4f}, {mei_np.max():.4f}]")
        print(f"  MEI L2 norm: {np.linalg.norm(mei_np):.4f}")
        print(f"{'='*60}")

    # ==========================================
    # 批量总结
    # ==========================================
    print(f"\n{'='*60}")
    print("Batch MEI Generation Complete")
    print(f"{'='*60}")
    print(f"  Processed neurons: {TARGET_NEURONS}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
