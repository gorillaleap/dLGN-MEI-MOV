"""
Training Script for Deep Retina Spatiotemporal Model - Spike Prediction

Sequence-to-Sequence 训练流程:
- 预测每个 Trial 的 23 帧连续 Spike/Firing Rate
- 使用 PoissonNLLLoss (适用于 Spike 数据)
- 评估指标: PSTH Pearson 相关系数

Usage:
    python train.py --data_path my_data.mat --n_neurons 50 --epochs 100

Author: Claude Code
Date: 2026-03-09
"""

import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import numpy as np
from tqdm import tqdm

# 导入自定义模块
from model import DeepRetinaSpatiotemporal, create_model
from dataset import CalciumImagingDataset, SpikeDataset, create_dataloaders
from utils import (
    PoissonLoss,
    poisson_nll_loss,
    pearson_correlation,
    frame_wise_pearson,
    psth_pearson,
    compute_spike_metrics,
    laplacian_penalty_3d,
    l1_regularization,
    SAM,
    get_first_conv_weight,
    save_checkpoint,
    load_checkpoint
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Train Deep Retina Spatiotemporal Model for Spike Prediction'
    )

    # 数据参数
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to data file (.mat, .npy, .h5)')
    parser.add_argument('--n_neurons', type=int, default=50,
                        help='Number of neurons to predict')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Training data ratio')

    # 模型参数
    parser.add_argument('--time_frames', type=int, default=23,
                        help='Number of time frames')
    parser.add_argument('--rf_diameter', type=int, default=80,
                        help='Receptive field diameter (pixels)')
    parser.add_argument('--n_channels', type=int, default=8,
                        help='Number of convolution channels')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization)')

    # SAM 优化器
    parser.add_argument('--use_sam', action='store_true',
                        help='Use SAM optimizer')
    parser.add_argument('--sam_rho', type=float, default=0.05,
                        help='SAM rho parameter')

    # 正则化
    parser.add_argument('--laplacian_weight', type=float, default=1e-5,
                        help='Laplacian regularization weight')
    parser.add_argument('--l1_weight', type=float, default=1e-5,
                        help='L1 regularization weight')

    # 其他
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--use_swanlab', action='store_true',
                        help='Use SwanLab for experiment tracking')

    return parser.parse_args()

class WeightedPoissonLoss(nn.Module):
    def __init__(self, log_input=False, full=True, alpha=5.0):
        """
        加权泊松损失函数
        alpha: 大值的放大系数。
               如果 target 是 0（基线），权重就是 1.0；
               如果 target 是 1.2（大峰值），权重就是 1.0 + 5.0 * 1.2 = 7.0 倍！
        """
        super().__init__()
        # 🚀 关键：reduction='none' 让它返回和 target 一样形状的 loss 矩阵，而不是一个标量
        self.poisson = nn.PoissonNLLLoss(log_input=log_input, full=full, reduction='none')
        self.alpha = alpha

    def forward(self, pred, target):
        # 1. 算基础的 Poisson Loss (逐元素)
        base_loss = self.poisson(pred, target)
        
        # 2. 构造动态权重矩阵：真实 target 越大，这块的 Loss 惩罚就越重
        # 加 1.0 是为了保证 baseline 至少有 1 倍的基础梯度，防止网络彻底不管基线
        weight = 1.0 + self.alpha * target
        
        # 3. 施加权重并求均值返回
        weighted_loss = torch.mean(base_loss * weight)
        
        return weighted_loss

def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batch_pearson_correlation(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    计算 Batch 内的均值 Pearson Correlation (可微分)

    解决 "MSE 偷懒陷阱": 模型预测过度平滑基线导致低 MSE 但低 Pearson。
    将 Pearson 直接加入 Loss，强迫模型注重时间序列形状。

    Args:
        preds: (B, T, N) 预测值 - B=batch, T=time, N=neurons
        targets: (B, T, N) 真实值
        eps: 防止除零

    Returns:
        pearson_r: 标量，均值 Pearson 相关系数 (范围约 [-1, 1])
    """
    # 在 Time 维度上计算均值
    pred_mean = preds.mean(dim=1, keepdim=True)  # (B, 1, N)
    target_mean = targets.mean(dim=1, keepdim=True)

    # 中心化
    pred_centered = preds - pred_mean  # (B, T, N)
    target_centered = targets - target_mean

    # 协方差 (在 Time 维度上)
    covariance = (pred_centered * target_centered).mean(dim=1)  # (B, N)

    # 标准差 (在 Time 维度上)
    pred_std = torch.sqrt((pred_centered ** 2).mean(dim=1) + eps)  # (B, N)
    target_std = torch.sqrt((target_centered ** 2).mean(dim=1) + eps)

    # Pearson 相关系数
    pearson_r = covariance / (pred_std * target_std)  # (B, N)

    # 返回全局均值
    return pearson_r.mean()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args,
    use_sam: bool = False
) -> dict:
    """
    训练一个 epoch - Poisson NLL Loss for Spike Rate Prediction

    关键变更 (Rollback):
    - 模型返回单一 tensor (B, 23, N_neurons)，非 tuple
    - 使用 PoissonNLLLoss 替代 MSE + Pearson + L1
    - 无需 tuple 解包

    Args:
        model: 模型 (返回 (B, 23, N_neurons) tensor)
        dataloader: 训练数据加载器
        optimizer: 优化器 (可能是 SAM 包装)
        device: 设备
        args: 参数对象 (需包含 laplacian_weight)
        use_sam: 是否使用 SAM 优化器

    Returns:
        dict: {'loss': float, 'poisson_nll': float, 'psth_pearson_r': float}
    """
    model.train()

    # Poisson NLL Loss for spike rate prediction
    # criterion = nn.PoissonNLLLoss(log_input=False, full=True, reduction='mean')
    criterion = WeightedPoissonLoss(log_input=False, full=True, alpha=8.0)

    total_loss = 0.0
    total_poisson = 0.0
    all_preds = []
    all_targets = []

    for batch_idx, (stimulus, behavior, response) in enumerate(dataloader):
        # 移到设备
        stimulus = stimulus.to(device)      # (B, 1, 23, 80, 80)
        behavior = behavior.to(device)      # (B, 23, 2)
        response = response.to(device)      # (B, 23, N_neurons) - CASCADE spikes

        if use_sam:
            # ==========================================
            # SAM Step 1: Forward + backward
            # ==========================================
            optimizer.zero_grad()

            # Model returns single tensor (B, 23, N_neurons)
            # Note: model output is already 23 frames
            predicted_spikes = model(stimulus, behavior)

            # --- Dimension Guard Wall ---
            if predicted_spikes.size(1) != 23 or response.size(1) != 23:
                raise ValueError(
                    f"[SHAPE MISMATCH] Preds: {predicted_spikes.shape}, Response: {response.shape}. "
                    f"Expected time dim = 23 for both!"
                )
            # ---------------------------

            # Poisson NLL Loss
            poisson_loss = criterion(predicted_spikes, response)

            # Add Laplacian regularization on first conv
            conv_weight = get_first_conv_weight(model)
            if conv_weight is not None and conv_weight.dim() == 5:
                loss = poisson_loss + args.laplacian_weight * laplacian_penalty_3d(conv_weight)
            else:
                loss = poisson_loss

            loss.backward()
            optimizer.first_step(zero_grad=True)

            # ==========================================
            # SAM Step 2: Forward again + update
            # ==========================================
            predicted_spikes = model(stimulus, behavior)

            # --- Dimension Guard Wall ---
            if predicted_spikes.size(1) != 23 or response.size(1) != 23:
                raise ValueError(
                    f"[SHAPE MISMATCH] Preds: {predicted_spikes.shape}, Response: {response.shape}. "
                    f"Expected time dim = 23 for both!"
                )
            # ---------------------------

            poisson_loss = criterion(predicted_spikes, response)

            # Add Laplacian regularization
            if conv_weight is not None and conv_weight.dim() == 5:
                loss = poisson_loss + args.laplacian_weight * laplacian_penalty_3d(conv_weight)
            else:
                loss = poisson_loss

            loss.backward()
            optimizer.second_step(zero_grad=True)

        else:
            # ==========================================
            # Standard Optimizer
            # ==========================================
            optimizer.zero_grad()

            # Model returns single tensor
            predicted_spikes = model(stimulus, behavior)

            # --- Dimension Guard Wall ---
            if predicted_spikes.size(1) != 23 or response.size(1) != 23:
                raise ValueError(
                    f"[SHAPE MISMATCH] Preds: {predicted_spikes.shape}, Response: {response.shape}. "
                    f"Expected time dim = 23 for both!"
                )
            # ---------------------------

            # Poisson NLL Loss
            poisson_loss = criterion(predicted_spikes, response)

            # Add Laplacian regularization
            conv_weight = get_first_conv_weight(model)
            if conv_weight is not None and conv_weight.dim() == 5:
                loss = poisson_loss + args.laplacian_weight * laplacian_penalty_3d(conv_weight)
            else:
                loss = poisson_loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # 累积统计
        total_loss += loss.item()
        total_poisson += poisson_loss.item()

        # 收集预测结果
        all_preds.append(predicted_spikes.detach())
        all_targets.append(response.detach())

    # 计算全局统计
    all_preds = torch.cat(all_preds, dim=0)    # (N_samples, 23, N_neurons)
    all_targets = torch.cat(all_targets, dim=0)

    # PSTH Pearson (平滑后)
    psth_r, _ = psth_pearson(all_preds, all_targets, kernel_size=3)

    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'poisson_nll': total_poisson / n_batches,
        'psth_pearson_r': psth_r.item()
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> dict:
    """
    验证模型 - Poisson NLL Loss 版本

    关键变更 (Rollback):
    - 模型返回单一 tensor (B, 23, N_neurons)，非 tuple
    - 使用 PoissonNLLLoss
    - 无需 tuple 解包

    Args:
        model: 模型 (返回 (B, 23, N_neurons) tensor)
        dataloader: 验证数据加载器
        device: 设备

    Returns:
        dict: {'loss', 'poisson_nll', 'psth_pearson_r', 'global_pearson_r', 'frame_pearson_r'}
    """
    model.eval()

    # Poisson NLL Loss for spike rate prediction
    criterion = nn.PoissonNLLLoss(log_input=False, full=True, reduction='mean')

    total_loss = 0.0
    total_poisson = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for stimulus, behavior, response in dataloader:
            stimulus = stimulus.to(device)
            behavior = behavior.to(device)
            response = response.to(device)

            # Model returns single tensor (B, 23, N_neurons)
            predicted_spikes = model(stimulus, behavior)

            # --- Dimension Guard Wall ---
            if predicted_spikes.size(1) != 23 or response.size(1) != 23:
                raise ValueError(
                    f"[SHAPE MISMATCH] Preds: {predicted_spikes.shape}, Response: {response.shape}. "
                    f"Expected time dim = 23 for both!"
                )
            # ---------------------------

            # Poisson NLL Loss
            poisson_loss = criterion(predicted_spikes, response)

            total_loss += poisson_loss.item()
            total_poisson += poisson_loss.item()

            all_preds.append(predicted_spikes)
            all_targets.append(response)

    # Global metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # PSTH Pearson
    psth_r, _ = psth_pearson(all_preds, all_targets, kernel_size=3)

    # Additional metrics from utils
    metrics = compute_spike_metrics(all_preds, all_targets)

    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'poisson_nll': total_poisson / n_batches,
        'psth_pearson_r': psth_r.item(),
        'global_pearson_r': metrics['global_pearson_r'],
        'frame_pearson_r': metrics['frame_pearson_r']
    }


def main():
    """主训练函数"""
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # SwanLab 初始化（可选）
    # ==========================================
    if args.use_swanlab:
        try:
            import swanlab
            swanlab.init(
                project="Deep-Retina-Spike-Prediction",
                experiment_name=f"seq2seq_{args.n_neurons}neurons_{args.epochs}epochs",
                config=vars(args)
            )
        except ImportError:
            print("Warning: swanlab not installed. Skipping experiment tracking.")
            args.use_swanlab = False

    # ==========================================
    # 数据加载
    # ==========================================
    print("\nLoading data...")

    if args.data_path is not None and Path(args.data_path).exists():
        # 加载真实数据
        dataset = CalciumImagingDataset(
            stimulus=args.data_path,
            behavior=None,  # 假设数据在同一文件
            response=None,
            rf_diameter=args.rf_diameter,
            is_training=True
        )
        args.n_neurons = dataset.n_neurons
    else:
        # 使用模拟 Spike 数据
        print("Using simulated spike data for testing...")
        dataset = SpikeDataset(
            n_trials=1000,
            n_neurons=args.n_neurons,
            time_frames=args.time_frames,
            rf_diameter=args.rf_diameter
        )

    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(
        dataset,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # ==========================================
    # 模型创建
    # ==========================================
    print("\nCreating model...")

    model = create_model(
        n_neurons=args.n_neurons,
        n_channels=args.n_channels,
        input_shape=(args.time_frames, args.rf_diameter, args.rf_diameter),
        device=device
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ==========================================
    # 损失函数: Poisson NLL (适用于 Spike 数据)
    # ==========================================
    criterion = PoissonLoss(log_input=False, full=True)
    print("Using PoissonNLLLoss for spike prediction")

    # ==========================================
    # 优化器
    # ==========================================
    if args.use_sam:
        base_optimizer = torch.optim.Adam
        optimizer = SAM(
            model.parameters(),
            base_optimizer=base_optimizer,
            rho=args.sam_rho
        )
        print(f"Using SAM optimizer with rho={args.sam_rho}")
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        print(f"Using Adam optimizer with lr={args.lr}")

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer if not args.use_sam else optimizer.base_optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    # ==========================================
    # 训练循环
    # ==========================================
    print("\nStarting training...")
    best_val_r = -float('inf')

    for epoch in range(args.epochs):
        # 训练
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            args=args,
            use_sam=args.use_sam
        )

        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            device=device
        )

        # 更新学习率
        scheduler.step()

        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Train R: {train_metrics['psth_pearson_r']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val R: {val_metrics['psth_pearson_r']:.4f}")

        # SwanLab 日志
        if args.use_swanlab:
            swanlab.log({
                'train/loss': train_metrics['loss'],
                'train/psth_pearson_r': train_metrics['psth_pearson_r'],
                'val/loss': val_metrics['loss'],
                'val/psth_pearson_r': val_metrics['psth_pearson_r'],
                'val/global_pearson_r': val_metrics['global_pearson_r'],
                'val/frame_pearson_r': val_metrics['frame_pearson_r'],
                'val/poisson_nll': val_metrics['poisson_nll'],
                'lr': scheduler.get_last_lr()[0]
            })

        # 保存最佳模型
        if val_metrics['psth_pearson_r'] > best_val_r:
            best_val_r = val_metrics['psth_pearson_r']
            save_checkpoint(
                model,
                optimizer if not args.use_sam else optimizer.base_optimizer,
                epoch,
                val_metrics,
                str(save_dir / 'best_model.pth')
            )

    # ==========================================
    # 训练结束
    # ==========================================
    print(f"\nTraining completed!")
    print(f"Best validation PSTH Pearson R: {best_val_r:.4f}")
    print(f"Best model saved to: {save_dir / 'best_model.pth'}")

    # 保存最终模型
    save_checkpoint(
        model,
        optimizer if not args.use_sam else optimizer.base_optimizer,
        args.epochs - 1,
        val_metrics,
        str(save_dir / 'final_model.pth')
    )

    if args.use_swanlab:
        swanlab.finish()


if __name__ == "__main__":
    main()
