"""
训练启动脚本 - Deep Retina Spatiotemporal Model
端到端钙前向模型版本

Usage:
    python run_training.py --data_path training_data.mat --epochs 100
"""

import torch
from pathlib import Path
import time        # ETA 时间计算
import swanlab     # 实验跟踪
import matplotlib.pyplot as plt
import numpy as np

from dataset import load_training_dataset, create_dataloaders
from model import create_model
from utils import SAM  # SAM 优化器
from train import train_one_epoch, validate, set_seed


def format_time(seconds: float) -> str:
    """将秒数格式化为易读的时间字符串"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def visualize_sanity_check(
    preds: torch.Tensor,
    targets: torch.Tensor,
    epoch: int,
    neuron_idx: int = 0,
    use_swanlab: bool = True,
    save_dir: Path = None
):
    """可视化预测 vs 真实值，用于检测数据错位"""
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    if preds.ndim == 3:
        pred_trace = preds[0, :, neuron_idx]
        target_trace = targets[0, :, neuron_idx]
    elif preds.ndim == 2:
        pred_trace = preds[:10, neuron_idx]
        target_trace = targets[:10, neuron_idx]
    else:
        raise ValueError(f"Unexpected preds shape: {preds.shape}")

    time_frames = len(pred_trace)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(range(time_frames), target_trace, 'b-o', label='True dF/F', linewidth=2, markersize=6)
    ax1.plot(range(time_frames), pred_trace, 'r--s', label='Predicted dF/F', linewidth=2, markersize=6)
    ax1.set_xlabel('Time Frame', fontsize=12)
    ax1.set_ylabel('dF/F', fontsize=12)
    ax1.set_title(f'Neuron {neuron_idx} - Epoch {epoch}', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.scatter(target_trace, pred_trace, c='blue', alpha=0.7, s=80, edgecolors='black')
    max_val = max(target_trace.max(), pred_trace.max()) + 0.1
    min_val = min(target_trace.min(), pred_trace.min()) - 0.1
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction', linewidth=2)
    ax2.set_xlabel('True dF/F', fontsize=12)
    ax2.set_ylabel('Predicted dF/F', fontsize=12)
    ax2.set_title('Prediction vs Target', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f'sanity_check_epoch{epoch}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [Sanity Check] Saved to {save_path}")

    if use_swanlab:
        try:
            swanlab.log({"val/sanity_check": swanlab.Image(fig)}, step=epoch)
            print(f"  [Sanity Check] Logged to SwanLab at epoch {epoch}")
        except Exception as e:
            print(f"  [Sanity Check] Failed to log to SwanLab: {e}")

    plt.close(fig)
    return fig


def main():
    # ==========================================
    # 配置参数
    # ==========================================
    DATA_PATH = "training_data4.mat"
    EXPAND_FACTOR = 30
    N_NEURONS = 50
    N_CHANNELS = 32
    RF_DIAMETER = 80
    BATCH_SIZE = 52
    EPOCHS = 25
    LR = 5e-5
    SEED = 42

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    set_seed(SEED)

    # ==========================================
    # Args 配置
    # ==========================================
    class Args:
        laplacian_weight = 1e-6
        l1_weight = 1e-6         # 也用作 sparsity_weight
        pearson_weight = 0.5     # Pearson 相关性权重 (防止 MSE 偷懒陷阱)
        use_sam = False # 现在是false用来调试
        sam_rho = 0.05
    args = Args()

    # ==========================================
    # SwanLab 初始化
    # ==========================================
    config = {
        "data_path": DATA_PATH,
        "expand_factor": EXPAND_FACTOR,
        "n_neurons": N_NEURONS,
        "n_channels": N_CHANNELS,
        "rf_diameter": RF_DIAMETER,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "seed": SEED,
        "optimizer": "SAM-Adam" if args.use_sam else "Adam",
        "sam_rho": args.sam_rho if args.use_sam else None,
        "laplacian_weight": args.laplacian_weight,
        "l1_weight": args.l1_weight,
        "pearson_weight": args.pearson_weight,
        "weight_decay": 1e-6,
        "scheduler": "CosineAnnealingLR",
        "loss": "MSE + L1 Sparsity - Pearson Correlation",
    }

    swanlab.init(
        project="MEI-MOV-DeepRetina",
        experiment_name=f"lr{LR}_bs{BATCH_SIZE}_ep{EPOCHS}",
        config=config
    )

    # ==========================================
    # 1. 加载数据
    # ==========================================
    print("\n" + "=" * 60)
    print("Step 1: Loading Data")
    print("=" * 60)

    dataset = load_training_dataset(
            mat_path=DATA_PATH,
            response_path='cascade_spikes_23frames.npy', # 🚀 核心改动：精准制导，直接读取预处理的 23 帧 Spike
            expand_factor=EXPAND_FACTOR,
            rf_diameter=RF_DIAMETER,
            is_training=True,
            normalize=True,
            use_cascade=False                            # 必须为 False，让 GPU 专心训练
        )

    train_loader, val_loader = create_dataloaders(
        dataset,
        train_ratio=0.8,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=True
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # ==========================================
    # 2. 创建模型
    # ==========================================
    print("\n" + "=" * 60)
    print("Step 2: Creating Model")
    print("=" * 60)

    model = create_model(
        n_neurons=dataset.n_neurons,
        n_channels=N_CHANNELS,
        input_shape=(dataset.time_frames, RF_DIAMETER, RF_DIAMETER),
        device=device
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # ==========================================
    # 3. 优化器 (不再需要 criterion)
    # ==========================================
    print("\n" + "=" * 60)
    print("Step 3: Setting Up Training")
    print("=" * 60)

    print("Loss: MSE + L1 Sparsity (internal, end-to-end)")

    if args.use_sam:
        optimizer = SAM(
            model.parameters(),
            base_optimizer=torch.optim.Adam,
            lr=LR,
            weight_decay=1e-4,
            rho=args.sam_rho
        )
        print(f"Optimizer: SAM-Adam (lr={LR}, rho={args.sam_rho})")
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LR,
            weight_decay=1e-4
        )
        print(f"Optimizer: Adam (lr={LR})")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer.base_optimizer if args.use_sam else optimizer,
        T_max=EPOCHS,
        eta_min=1e-6
    )
    print(f"Scheduler: CosineAnnealingLR")

    # ==========================================
    # 4. 训练循环
    # ==========================================
    print("\n" + "=" * 60)
    print("Step 4: Training")
    print("=" * 60)

    save_dir = Path('./checkpoints')
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_r = -float('inf')
    training_start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()

        # ==========================================
        # 训练 (使用关键字参数，移除 criterion)
        # ==========================================
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            args=args,
            use_sam=args.use_sam
        )

        # ==========================================
        # 验证 (使用关键字参数，移除 criterion)
        # ==========================================
        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            device=device
        )


        # ==========================================
        # 每 Epoch 可视化 Sanity Check 并上传 SwanLab
        # ==========================================
        model.eval()
        with torch.no_grad():
            for batch_stim, batch_beh, batch_resp in val_loader:
                batch_stim = batch_stim.to(device)
                batch_beh = batch_beh.to(device)
                batch_resp = batch_resp.to(device)

                # 模型返回单一 tensor (B, 23, N_neurons)
                predicted_spikes = model(batch_stim, batch_beh)

                # 可视化：保存到本地并自动推送到 SwanLab
                visualize_sanity_check(
                    preds=predicted_spikes,
                    targets=batch_resp,
                    epoch=epoch,
                    neuron_idx=0,
                    use_swanlab=True,
                    save_dir=save_dir
                )
                break  # 极其重要：只取第一个 validation batch 画一张图就退出循环

        # 更新学习率
        scheduler.step()

        # ==========================================
        # ETA 计算
        # ==========================================
        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - training_start_time
        avg_epoch_time = elapsed_time / (epoch + 1)
        remaining_epochs = EPOCHS - (epoch + 1)
        eta_seconds = avg_epoch_time * remaining_epochs

        # ==========================================
        # SwanLab 记录指标
        # ==========================================
        swanlab.log({
            "train/loss": train_metrics['loss'],
            "train/poisson_nll": train_metrics.get('poisson_nll', train_metrics['loss']),
            "train/pearson_r": train_metrics['psth_pearson_r'],
            "val/loss": val_metrics['loss'],
            "val/poisson_nll": val_metrics.get('poisson_nll', val_metrics['loss']),
            "val/pearson_r": val_metrics['psth_pearson_r'],
            "val/global_pearson_r": val_metrics.get('global_pearson_r', 0),
            "val/frame_pearson_r": val_metrics.get('frame_pearson_r', 0),
            "train/lr": scheduler.get_last_lr()[0],
            "train/epoch_time": epoch_time,
        })

        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[Epoch {epoch+1:3d}/{EPOCHS}] "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val R: {val_metrics['psth_pearson_r']:.4f} | "
                  f"Time: {format_time(epoch_time)} | "
                  f"ETA: {format_time(eta_seconds)}")

        # 保存最佳模型
        if val_metrics['psth_pearson_r'] > best_val_r:
            best_val_r = val_metrics['psth_pearson_r']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_r': best_val_r
            }, save_dir / 'best_model.pth')

    # ==========================================
    # 5. 训练完成
    # ==========================================
    total_training_time = time.time() - training_start_time
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best validation PSTH Pearson R: {best_val_r:.4f}")
    print(f"Total training time: {format_time(total_training_time)}")
    print(f"Best model saved to: {save_dir / 'best_model.pth'}")

    swanlab.finish()


if __name__ == "__main__":
    main()
