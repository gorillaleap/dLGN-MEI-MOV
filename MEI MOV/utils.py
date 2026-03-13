"""
Utility Functions for Deep Retina Spatiotemporal Model - Spike Prediction

包含:
- Poisson NLL Loss (用于 Spike/Firing Rate 预测)
- PSTH Pearson 相关系数 (平滑后逐帧评估)
- 逐帧 Pearson 相关系数
- SAM 优化器
- Laplacian 正则化
- 其他训练和评估工具

Author: Claude Code
Date: 2026-03-09
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import numpy as np


# ==========================================
# 损失函数
# ==========================================

class PoissonLoss(nn.Module):
    """
    Poisson Negative Log-Likelihood Loss

    专门用于 Spike/Firing Rate 预测

    Loss = λ - y * log(λ)  (其中 λ 是预测的 firing rate, y 是真实 spike count)

    Args:
        log_input: 如果 True，期望输入是 log(λ)；如果 False，期望输入是 λ
        full: 是否包含 Stirling 近似项
        eps: 数值稳定性常数
    """

    def __init__(self, log_input: bool = False, full: bool = True, eps: float = 1e-8):
        super().__init__()
        self.log_input = log_input
        self.full = full
        self.eps = eps
        self.criterion = nn.PoissonNLLLoss(
            log_input=log_input,
            full=full,
            reduction='mean'
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (Batch, 23, N_neurons) - 预测的 Firing Rate (Softplus 后, ≥ 0)
            target: (Batch, 23, N_neurons) - 真实的 Spike Count (≥ 0)

        Returns:
            loss: scalar
        """
        # 确保预测值非负 (防止数值问题)
        pred = torch.clamp(pred, min=self.eps)
        target = torch.clamp(target, min=0)

        return self.criterion(pred, target)


def poisson_nll_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    log_input: bool = False,
    full: bool = True
) -> torch.Tensor:
    """
    函数式 Poisson NLL Loss

    Args:
        pred: (Batch, 23, N_neurons) - 预测的 Firing Rate
        target: (Batch, 23, N_neurons) - 真实的 Spike Count

    Returns:
        loss: scalar
    """
    return F.poisson_nll_loss(pred, target, log_input=log_input, full=full, reduction='mean')


# ==========================================
# 钙前向模型损失函数 (新增)
# ==========================================

class CalciumForwardLoss(nn.Module):
    """
    端到端钙信号预测损失函数

    用于 CalciumForwardModel 端到端训练。
    真实标签是连续的钙荧光信号 (dF/F)，而非去卷积后的脉冲。

    组成:
    ----
    1. Main Loss: predicted_dff vs ground_truth_dff 的 MSE
       - 监督模型学习正确的钙荧光动态
    2. Sparsity Loss: latent_spikes 的 L1 范数
       - 强制网络学习稀疏的脉冲表征 (类似 L1 正则化)
       - 避免网络输出连续的噪声而非离散的脉冲

    Total Loss = MSE + alpha * L1

    张量形状:
    --------
    - predicted_dff: (B, Time, N_neurons) - 模型预测的钙荧光
    - ground_truth_dff: (B, Time, N_neurons) - 真实的 dF/F 信号
    - latent_spikes: (B, Time, N_neurons) - 内部稀疏脉冲表征

    Args:
        alpha: 稀疏性惩罚权重 (默认 0.01)
    """

    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(
        self,
        predicted_dff: torch.Tensor,
        ground_truth_dff: torch.Tensor,
        latent_spikes: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算总损失

        Args:
            predicted_dff: 预测的钙信号 (B, T, N)
            ground_truth_dff: 真实的钙信号 (B, T, N)
            latent_spikes: 潜在脉冲 (B, T, N)

        Returns:
            total_loss: 总损失 (标量)
            loss_dict: 各分量损失字典，用于日志记录
        """
        # Main Loss: 重建误差 (MSE)
        # 衡量预测的钙荧光与真实信号的相似度
        mse_loss = self.mse(predicted_dff, ground_truth_dff)

        # Sparsity Loss: L1 范数
        # 强制 latent_spikes 稀疏化 (大部分时间点为 0)
        # 这模拟了真实神经元的稀疏放电特性
        sparsity_loss = torch.mean(torch.abs(latent_spikes))

        # Total Loss
        total_loss = mse_loss + self.alpha * sparsity_loss

        # 返回损失字典，方便日志记录
        loss_dict = {
            'mse_loss': mse_loss.item(),
            'sparsity_loss': sparsity_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict


def calcium_forward_loss(
    predicted_dff: torch.Tensor,
    ground_truth_dff: torch.Tensor,
    latent_spikes: torch.Tensor,
    alpha: float = 0.01
) -> Tuple[torch.Tensor, dict]:
    """
    函数式钙前向损失

    Args:
        predicted_dff: (B, T, N) - 预测的钙荧光
        ground_truth_dff: (B, T, N) - 真实的 dF/F
        latent_spikes: (B, T, N) - 潜在脉冲
        alpha: 稀疏性权重

    Returns:
        total_loss, loss_dict
    """
    criterion = CalciumForwardLoss(alpha=alpha)
    return criterion(predicted_dff, ground_truth_dff, latent_spikes)


# ==========================================
# 评估指标
# ==========================================

def pearson_correlation(
    preds: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算 Pearson 相关系数 (全局统计版)

    Args:
        preds: (Batch, N) - 预测值
        targets: (Batch, N) - 真实值
        eps: 数值稳定性常数

    Returns:
        mean_r: 所有神经元的平均相关系数
        per_neuron_r: (N,) - 每个神经元的相关系数
    """
    if preds.dim() == 1:
        preds = preds.unsqueeze(1)
    if targets.dim() == 1:
        targets = targets.unsqueeze(1)

    # 计算均值
    pred_mean = torch.mean(preds, dim=0, keepdim=True)
    target_mean = torch.mean(targets, dim=0, keepdim=True)

    # 计算协方差
    cov = torch.mean((preds - pred_mean) * (targets - target_mean), dim=0)

    # 计算标准差
    pred_std = torch.std(preds, dim=0)
    target_std = torch.std(targets, dim=0)

    # 计算相关系数
    denominator = torch.clamp(pred_std * target_std, min=eps)
    per_neuron_r = cov / denominator

    # 限制在 [-1, 1] 范围内
    per_neuron_r = torch.clamp(per_neuron_r, -1.0, 1.0)

    mean_r = torch.mean(per_neuron_r)

    return mean_r, per_neuron_r


def frame_wise_pearson(
    preds: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    逐帧 Pearson 相关系数 (Sequence-to-Sequence 评估)

    对每一帧单独计算 Pearson 相关系数，然后平均

    Args:
        preds: (Batch, Time, N_neurons) - 预测的时间序列
        targets: (Batch, Time, N_neurons) - 真实的时间序列
        eps: 数值稳定性常数

    Returns:
        mean_r: 平均相关系数 (跨帧、跨神经元)
        per_frame_r: (Time,) - 每帧的平均相关系数
    """
    B, T, N = preds.shape

    per_frame_r = []
    for t in range(T):
        # 每帧: (Batch, N_neurons)
        pred_t = preds[:, t, :]  # (B, N)
        target_t = targets[:, t, :]  # (B, N)

        # 展平为 (B*N,) 进行全局统计
        pred_flat = pred_t.reshape(-1)
        target_flat = target_t.reshape(-1)

        # 计算该帧的 Pearson r
        r, _ = pearson_correlation(pred_flat, target_flat, eps)
        per_frame_r.append(r)

    per_frame_r = torch.stack(per_frame_r)  # (Time,)
    mean_r = torch.mean(per_frame_r)

    return mean_r, per_frame_r


def psth_pearson(
    preds: torch.Tensor,
    targets: torch.Tensor,
    kernel_size: int = 3,
    eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PSTH (Peri-Stimulus Time Histogram) 平滑后的 Pearson 相关系数

    1. 对预测和真实响应进行时间平滑 (Gaussian 或 Moving Average)
    2. 计算平滑后的逐帧 Pearson 相关系数

    Args:
        preds: (Batch, Time, N_neurons) - 预测的时间序列
        targets: (Batch, Time, N_neurons) - 真实的时间序列
        kernel_size: 平滑核大小 (奇数)
        eps: 数值稳定性常数

    Returns:
        mean_r: 平滑后的平均相关系数
        per_frame_r: (Time,) - 平滑后每帧的相关系数
    """
    B, T, N = preds.shape

    # ==========================================
    # Step 1: 时间平滑 (Moving Average)
    # ==========================================
    # 使用 1D 平均池化进行平滑
    # (Batch, Time, N) -> (Batch, N, Time) for Conv1d
    preds_t = preds.permute(0, 2, 1)  # (B, N, T)
    targets_t = targets.permute(0, 2, 1)  # (B, N, T)

    # 添加 channel 维度用于 AvgPool1d
    preds_t = preds_t.unsqueeze(1)  # (B, 1, N, T)
    targets_t = targets_t.unsqueeze(1)  # (B, 1, N, T)

    # 对每个 (B, n) 应用 1D 平均池化
    # 先 reshape 为 (B*N, 1, T)
    preds_flat = preds_t.reshape(B * N, 1, T)
    targets_flat = targets_t.reshape(B * N, 1, T)

    # 平滑
    padding = kernel_size // 2
    preds_smooth = F.avg_pool1d(preds_flat, kernel_size, stride=1, padding=padding)
    targets_smooth = F.avg_pool1d(targets_flat, kernel_size, stride=1, padding=padding)

    # 恢复形状
    preds_smooth = preds_smooth.reshape(B, N, T).permute(0, 2, 1)  # (B, T, N)
    targets_smooth = targets_smooth.reshape(B, N, T).permute(0, 2, 1)  # (B, T, N)

    # ==========================================
    # Step 2: 逐帧 Pearson
    # ==========================================
    return frame_wise_pearson(preds_smooth, targets_smooth, eps)


def compute_spike_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor
) -> dict:
    """
    计算 Spike 预测的所有评估指标

    Args:
        preds: (Batch, Time, N_neurons) - 预测的 Firing Rate
        targets: (Batch, Time, N_neurons) - 真实的 Spike Count

    Returns:
        metrics: dict
    """
    # 确保非负
    preds = torch.clamp(preds, min=1e-8)
    targets = torch.clamp(targets, min=0)

    # 展平为 (Batch*Time, N_neurons) 计算全局指标
    B, T, N = preds.shape
    preds_flat = preds.reshape(-1, N)
    targets_flat = targets.reshape(-1, N)

    # 全局 Pearson
    global_r, per_neuron_r = pearson_correlation(preds_flat, targets_flat)

    # 逐帧 Pearson
    frame_r, per_frame_r = frame_wise_pearson(preds, targets)

    # PSTH Pearson (平滑后)
    psth_r, psth_per_frame = psth_pearson(preds, targets, kernel_size=3)

    # Poisson NLL
    poisson_loss = poisson_nll_loss(preds, targets, log_input=False, full=True)

    # MSE (用于参考)
    mse = F.mse_loss(preds, targets)

    # MAE
    mae = F.l1_loss(preds, targets)

    return {
        'global_pearson_r': global_r.item(),
        'per_neuron_r': per_neuron_r.detach(),
        'frame_pearson_r': frame_r.item(),
        'per_frame_r': per_frame_r.detach(),
        'psth_pearson_r': psth_r.item(),
        'psth_per_frame': psth_per_frame.detach(),
        'poisson_nll': poisson_loss.item(),
        'mse': mse.item(),
        'mae': mae.item()
    }


# ==========================================
# 正则化
# ==========================================

def laplacian_penalty_3d(weight: torch.Tensor) -> torch.Tensor:
    """
    Laplacian 正则化 (3D 卷积核版本)

    惩罚高频波动，鼓励平滑的时空卷积核

    Args:
        weight: (out_ch, in_ch, T, H, W) - 3D 卷积权重

    Returns:
        penalty: 标量惩罚值
    """
    if weight is None or weight.dim() != 5:
        return torch.tensor(0.0, device=weight.device if weight is not None else 'cuda')

    # 3D Laplacian 核
    lap_kernel_3d = torch.tensor(
        [[[[[0, 0, 0],
            [0, -1, 0],
            [0, 0, 0]],
           [[0, -1, 0],
            [-1, 6, -1],
            [0, -1, 0]],
           [[0, 0, 0],
            [0, -1, 0],
            [0, 0, 0]]]]],
        dtype=weight.dtype,
        device=weight.device
    ).reshape(1, 1, 3, 3, 3)

    # 对每个输出通道应用
    weight_reshaped = weight.view(-1, 1, *weight.shape[2:])  # (out*in, 1, T, H, W)
    lap_maps = F.conv3d(weight_reshaped, lap_kernel_3d, padding=1)

    return torch.sum(lap_maps ** 2)


def laplacian_penalty_2d(weight: torch.Tensor) -> torch.Tensor:
    """
    Laplacian 正则化 (2D 卷积核版本)

    Args:
        weight: (out_ch, in_ch, H, W) - 2D 卷积权重

    Returns:
        penalty: 标量惩罚值
    """
    if weight is None or weight.dim() != 4:
        return torch.tensor(0.0, device=weight.device if weight is not None else 'cuda')

    lap_kernel = torch.tensor(
        [[0, -1, 0],
         [-1, 4, -1],
         [0, -1, 0]],
        dtype=weight.dtype,
        device=weight.device
    )
    lap_kernel = lap_kernel.view(1, 1, 3, 3)
    lap_kernel = lap_kernel.repeat(weight.size(1), 1, 1, 1)

    weight_reshaped = weight.view(-1, weight.size(2), weight.size(3)).unsqueeze(1)
    lap_maps = F.conv2d(weight_reshaped, lap_kernel, padding=1)

    return torch.sum(lap_maps ** 2)


def l1_regularization(model: nn.Module) -> torch.Tensor:
    """计算模型的 L1 正则化"""
    l1_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for param in model.parameters():
        l1_loss = l1_loss + torch.sum(torch.abs(param))
    return l1_loss


def l2_regularization(model: nn.Module) -> torch.Tensor:
    """计算模型的 L2 正则化"""
    l2_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for param in model.parameters():
        l2_loss = l2_loss + torch.sum(param ** 2)
    return l2_loss


# ==========================================
# SAM 优化器
# ==========================================

class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) 优化器

    论文: "Sharpness-Aware Minimization for Efficiently Improving Generalization"
    https://arxiv.org/abs/2010.01412

    用法:
        optimizer = SAM(model.parameters(), base_optimizer=torch.optim.Adam, rho=0.05)

        # 前向传播 1
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        # 前向传播 2
        criterion(model(x), y).backward()
        optimizer.second_step(zero_grad=True)
    """

    def __init__(self, params, base_optimizer, rho: float = 0.05, adaptive: bool = False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        # 把 lr, weight_decay 等额外的参数吸收进来
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """第一步：计算梯度并扰动参数"""
        grad_norm = self._grad_norm()

        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)

            for p in group['params']:
                if p.grad is None:
                    continue

                if p.dim() > 1:  # 只对非 bias 参数扰动
                    e_w = p.grad * scale.to(p)
                    self.state[p]['e_w'] = e_w
                    p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """第二步：恢复参数并更新"""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if 'e_w' in state:
                    p.sub_(state['e_w'])
                    del state['e_w']

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    def _grad_norm(self) -> torch.Tensor:
        """计算梯度范数"""
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.zeros([], device=shared_device)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    norm += p.grad.data.norm(2).to(shared_device) ** 2

        return norm.sqrt()

    @torch.no_grad()
    def step(self, closure=None):
        """标准 step 方法（SAM 需要两步操作，不建议直接使用）"""
        raise NotImplementedError("SAM requires first_step() and second_step(). See class docstring.")


# ==========================================
# 模型工具
# ==========================================

def get_first_conv_weight(model: nn.Module) -> Optional[torch.Tensor]:
    """获取模型的第一个卷积层权重（用于正则化）"""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            return m.weight
    return None


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """计算模型参数数量"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: str
):
    """保存训练检查点"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
    device: torch.device
) -> dict:
    """加载训���检查点"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint


# ==========================================
# 测试代码
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Utility Functions for Spike Prediction")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # ==========================================
    # Test 1: Poisson Loss
    # ==========================================
    print("\n[1] Testing PoissonLoss:")
    criterion = PoissonLoss(log_input=False, full=True)

    # 模拟预测和真实 spike
    batch_size, time_frames, n_neurons = 32, 23, 50
    preds = torch.relu(torch.randn(batch_size, time_frames, n_neurons, device=device)) + 0.5
    targets = torch.relu(torch.randn(batch_size, time_frames, n_neurons, device=device))

    loss = criterion(preds, targets)
    print(f"  Preds shape: {preds.shape}")
    print(f"  Targets shape: {targets.shape}")
    print(f"  Poisson NLL Loss: {loss.item():.4f}")

    # ==========================================
    # Test 2: Pearson Correlation
    # ==========================================
    print("\n[2] Testing Pearson Correlation:")
    mean_r, per_r = pearson_correlation(preds.reshape(-1, n_neurons), targets.reshape(-1, n_neurons))
    print(f"  Mean Pearson r: {mean_r.item():.4f}")
    print(f"  Per-neuron r shape: {per_r.shape}")

    # ==========================================
    # Test 3: Frame-wise Pearson
    # ==========================================
    print("\n[3] Testing Frame-wise Pearson:")
    frame_r, per_frame_r = frame_wise_pearson(preds, targets)
    print(f"  Mean frame r: {frame_r.item():.4f}")
    print(f"  Per-frame r shape: {per_frame_r.shape}")
    print(f"  Per-frame r (first 5): {per_frame_r[:5].tolist()}")

    # ==========================================
    # Test 4: PSTH Pearson
    # ==========================================
    print("\n[4] Testing PSTH Pearson (smoothed):")
    psth_r, psth_per_frame = psth_pearson(preds, targets, kernel_size=3)
    print(f"  Mean PSTH r: {psth_r.item():.4f}")
    print(f"  PSTH per-frame r shape: {psth_per_frame.shape}")

    # ==========================================
    # Test 5: Complete Metrics
    # ==========================================
    print("\n[5] Testing compute_spike_metrics:")
    metrics = compute_spike_metrics(preds, targets)
    for key, value in metrics.items():
        if not 'per' in key:
            print(f"  {key}: {value:.4f}")

    # ==========================================
    # Test 6: SAM Optimizer
    # ==========================================
    print("\n[6] Testing SAM optimizer:")
    model = nn.Linear(100, 50).to(device)
    base_opt = torch.optim.Adam
    optimizer = SAM(model.parameters(), base_optimizer=base_opt, rho=0.05)

    x = torch.randn(8, 100, device=device)
    y = torch.relu(torch.randn(8, 50, device=device))

    # Step 1
    output = model(x)
    loss = poisson_nll_loss(torch.relu(output) + 0.1, y, log_input=False)
    loss.backward()
    optimizer.first_step(zero_grad=True)
    print(f"  First step completed, loss: {loss.item():.4f}")

    # Step 2
    output2 = model(x)
    loss2 = poisson_nll_loss(torch.relu(output2) + 0.1, y, log_input=False)
    loss2.backward()
    optimizer.second_step(zero_grad=True)
    print(f"  Second step completed, loss: {loss2.item():.4f}")

    print("\n[OK] All utility tests passed!")
