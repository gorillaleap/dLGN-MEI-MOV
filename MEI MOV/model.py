"""
Deep Retina Spatiotemporal Model - Sequence-to-Sequence Architecture

Reference: Maheswaranathan et al., Neuron 2023
Stanford Baccus Lab - Retinal Neural Code for Natural Scenes

Adapted for two-photon calcium imaging with DECONVOLVED SPIKE data:
- Sequence-to-Sequence prediction (preserve 23 time frames)
- Output: Firing rates (must be non-negative via Softplus)
- Loss: Poisson Negative Log-Likelihood

Author: Claude Code
Date: 2026-03-09
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class BehaviorModulator(nn.Module):
    """
    时序行为调制器 (逐帧调制)

    输入: (Batch, 23, 2) - 行为时间序列
    输出: (Batch, 23, N_neurons) - 逐帧 Offset

    调制方式: y = base + offset (加性调制)
    """

    def __init__(self, behavior_dim: int = 2, n_neurons: int = 50, hidden_dim: int = 32):
        super().__init__()

        # 共享 MLP
        self.shared = nn.Sequential(
            nn.Linear(behavior_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 输出 Offset (可正可负)
        self.offset_head = nn.Linear(hidden_dim, n_neurons)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, behavior: torch.Tensor) -> torch.Tensor:
        """
        Args:
            behavior: (Batch, Time=23, 2)

        Returns:
            offset: (Batch, Time=23, N_neurons)
        """
        B, T, _ = behavior.shape
        # 展平批次和时间维度
        beh_flat = behavior.view(B * T, -1)  # (B*T, 2)

        # 通过 MLP
        h = self.shared(beh_flat)  # (B*T, hidden)
        offset = self.offset_head(h)  # (B*T, N_neurons)

        # 恢复时间维度
        return offset.view(B, T, -1)  # (B, 23, N_neurons)


class DeepRetina3D(nn.Module):
    """
    3D 卷积视网膜模型 + Factorized Readout + FiLM 行为调制

    参照 eLife 86860 (Höfling et al., 2024) ���宾根大学的方法，
    使用 Factorized Readout 替代全连接层，将空间和通道特征分离处理。

    时间维度推导 (无 temporal padding):
        33 - (5-1) = 29  (Layer 1: kernel=5)
        29 - (5-1) = 25  (Layer 2: kernel=5)
        25 - (3-1) = 23  (Layer 3: kernel=3)
    总时间感受野: 11 帧

    Gaussian Factorized Readout 优势:
    - 参数效率: 3,400 vs 原全连接 1,651,506 (减少 99.8%)
    - 空间连续性: 高斯掩码强制空间平滑，避免高频噪点
    - 可解释性: mu = RF 中心, sigma = RF 尺寸
    - 通道选择性: 每个神经元学习独立的 64 维特征权重
    """
    def __init__(
        self,
        n_neurons: int = 50,
        input_time: int = 33,
        input_height: int = 80,
        input_width: int = 80,
        behavior_dim: int = 2
    ):
        super().__init__()

        self.n_neurons = n_neurons
        self.input_time = input_time

        # ==========================================
        # Layer 1: Conv3d + Spatial Pool (80x80 -> 40x40)
        # 时间核: 5, 输入33帧 -> 输出29帧
        # ==========================================
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(5, 7, 7), stride=1, padding=(0, 3, 3))
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # ==========================================
        # Layer 2: Conv3d + Spatial Pool (40x40 -> 20x20)
        # 时间核: 5, 输入29帧 -> 输出25帧
        # ==========================================
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(5, 5, 5), stride=1, padding=(0, 2, 2))
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # ==========================================
        # Layer 3: Conv3d + Spatial Pool (20x20 -> 10x10) + 残差
        # 时间核: 3, 输入25帧 -> 输出23帧
        # ==========================================
        self.conv3 = nn.Conv3d(64, 64, kernel_size=(3, 5, 5), stride=1, padding=(0, 2, 2))
        self.bn3 = nn.BatchNorm3d(64)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # ==========================================
        # Gaussian Factorized Readout (核心创新)
        # ==========================================
        # 彻底解耦空间与特征权重，避免 Flatten 破坏空间拓扑
        # 这解决了 Flatten + Linear 读出层在 MEI 生成时产生高频对抗性噪点的问题

        # 1. 高斯空间掩码参数 (Spatial Parameters)
        # mu: 感受野中心坐标 (x, y), 范围 [-1, 1]
        self.mu = nn.Parameter(torch.zeros(n_neurons, 2))

        # sigma: 感受野尺寸 (允许学习不同尺寸的 RF)
        self.sigma = nn.Parameter(torch.ones(n_neurons, 1) * 0.5)

        # 2. 固定空间网格 (10x10), 注册为 buffer 防止设备不同步
        spatial_size = 10  # Layer 3 输出空间尺寸
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, spatial_size),
            torch.linspace(-1, 1, spatial_size),
            indexing='ij'
        )
        self.register_buffer('grid', torch.stack([grid_x, grid_y], dim=0))  # (2, 10, 10)

        # 3. 特征权重与偏置 (Feature Parameters)
        # feature_weights: 每个神经元的特征偏好向量 (64 通道 -> n_neurons)
        self.feature_weights = nn.Parameter(torch.randn(n_neurons, 64) * 0.01)

        # bias: 初始化为 -2.0 (契合生物学背景自发噪音)
        # softplus(-2.0) ≈ 0.12, 符合真实背景放电率
        self.bias = nn.Parameter(torch.full((n_neurons,), -2.0))

        # ==========================================
        # FiLM: Feature-wise Linear Modulation
        # 输出维度改为 64，匹配 feature_weights 维度
        # ==========================================
        self.gain_net = nn.Linear(behavior_dim, 64)
        self.shift_net = nn.Linear(behavior_dim, 64)

    def forward(self, stimulus: torch.Tensor, behavior: torch.Tensor = None) -> torch.Tensor:
        B = stimulus.shape[0]

        # ==========================================
        # Layer 1: (B, 1, 33, 80, 80) -> (B, 32, 29, 40, 40)
        # 时间核5: 33 - 4 = 29
        # ==========================================
        x = self.conv1(stimulus)
        x = F.softplus(self.bn1(x))
        x = self.pool1(x)

        # ==========================================
        # Layer 2: (B, 32, 29, 40, 40) -> (B, 64, 25, 20, 20)
        # 时间核5: 29 - 4 = 25
        # ==========================================
        x = self.conv2(x)
        x = F.softplus(self.bn2(x))
        x = self.pool2(x)

        # 保存残差 (用于 Layer 3)
        # x: (B, 64, 25, 20, 20)
        identity = x

        # ==========================================
        # Layer 3: (B, 64, 25, 20, 20) -> (B, 64, 23, 10, 10) + 残差
        # 时间核3: 25 - 2 = 23
        # ==========================================
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.softplus(x)
        x = self.pool3(x)  # (B, 64, 23, 10, 10)

        # ============================================================
        # 残差分支处理 - 因果律解释 (Causal Constraint)
        # ------------------------------------------------------------
        # identity 有 25 帧, 主分支输出 23 帧
        # 必须丢弃 identity 的前 2 帧, 保留后 23 帧
        #
        # 为什么丢弃前2帧?
        # - Layer 3 的卷积核大小为 3, 会消耗 2 帧的时间信息
        # - 主分支的 23 帧输出对应 identity 的后 23 帧 (第3-25帧)
        # - response[0] 依赖的是 identity[2:] 开始的刺激信息
        # - 丢弃前 2 帧确保残差与主分支在时间上对齐, 符合因果约束
        #
        # identity_cropped = identity[:, :, -23:, :, :]
        # 意味着: 保留最后 23 帧, 丢弃最早的 2 帧
        # ============================================================
        identity_cropped = identity[:, :, -23:, :, :]  # (B, 64, 23, 20, 20)
        identity_pooled = self.pool3(identity_cropped)  # (B, 64, 23, 10, 10)

        # 残差融合
        x = x + identity_pooled  # (B, 64, 23, 10, 10)

        # ==========================================
        # Gaussian Factorized Readout (核心创新)
        # ==========================================
        # 彻底解耦空间与特征权重，避免 Flatten 破坏空间拓扑
        # 这解决了 MEI 生成时的高频对抗性噪点（迷彩病）问题
        #
        # 输入 x: (B, C, T, H, W) = (B, 64, 23, 10, 10)
        #         │   │   │   │   └── 空间宽度
        #         │   │   │   └─────── 空间高度
        #         │   │   └─────────── 时间帧
        #         │   └─────────────── 通道数
        #         └─────────────────── Batch

        # Step 1: 动态渲染高斯空间掩码
        # 公式: mask = exp(-((X-μx)² + (Y-μy)²) / 2σ²)
        mu_x = self.mu[:, 0].view(-1, 1, 1)  # (n_neurons, 1, 1)
        mu_y = self.mu[:, 1].view(-1, 1, 1)  # (n_neurons, 1, 1)
        sigma_sq = (self.sigma ** 2).view(-1, 1, 1)  # (n_neurons, 1, 1)

        grid_x = self.grid[0].unsqueeze(0)  # (1, H, W)
        grid_y = self.grid[1].unsqueeze(0)  # (1, H, W)

        # 计算高斯
        spatial_mask = torch.exp(-(
            (grid_x - mu_x) ** 2 + (grid_y - mu_y) ** 2
        ) / (2 * sigma_sq + 1e-8))  # (n_neurons, H, W)

        # Softmax 归一化 (确保和为 1，变成软空间注意力)
        spatial_mask = F.softmax(spatial_mask.view(self.n_neurons, -1), dim=1)
        spatial_mask = spatial_mask.view(self.n_neurons, 10, 10)  # (n_neurons, 10, 10)

        # Step 2: 空间聚合 (Einsum Magic)
        # 'bcthw,nhw->bctn': 对 H,W 维度加权求和
        spatial_pooled = torch.einsum('bcthw,nhw->bctn', x, spatial_mask)
        # 结果: (B, 64, 23, n_neurons)

        # Step 3: FiLM 行为调制 (在通道维度上)
        if behavior is not None:
            # behavior: (B, 23, 2)
            # gain_net/shift_net: Linear(2 -> 64)
            gain_raw = self.gain_net(behavior)  # (B, 23, 64)
            shift = self.shift_net(behavior)     # (B, 23, 64)

            # 保证 gain >= 1.0，使用 softplus 确保非负
            gain = 1.0 + F.softplus(gain_raw)    # (B, 23, 64)

            # 调制: spatial_pooled 是 (B, C, T, N)，需要转置
            spatial_pooled = spatial_pooled.permute(0, 2, 1, 3)  # (B, 23, 64, N)
            spatial_pooled = spatial_pooled * gain.unsqueeze(-1) + shift.unsqueeze(-1)
            spatial_pooled = spatial_pooled.permute(0, 2, 1, 3)  # (B, 64, 23, N)

        # Step 4: 特征聚合 (Einsum Magic)
        # 'bctn,nc->btn': 对 C 维度加权求和
        base_response = torch.einsum('bctn,nc->btn', spatial_pooled, self.feature_weights)
        # 结果: (B, 23, n_neurons)

        # Step 5: 加偏置
        base_response = base_response + self.bias  # (B, 23, n_neurons)

        # ==========================================
        # Softplus 输出 (保证非负)
        # ==========================================
        firing_rate = F.softplus(base_response)  # (B, 23, n_neurons)

        return firing_rate

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DeepRetinaSpatiotemporal(nn.Module):
    """
    时空卷积视网膜模型 - Sequence-to-Sequence 架构

    专为 Deconvolved Spike 数据设计：
    - 输入: (B, 1, 23, 80, 80) - 23帧视频
    - 输出: (B, 23, N_neurons) - 逐帧 Firing Rate (Softplus 保证非负)

    架构概览:
    ┌─────────────────────────────────────────────────────────────┐
    │  Input: (B, 1, 23, 80, 80)                                 │
    │    ↓                                                        │
    │  Layer 1: Conv3d(5,21,21) + BN + SpatialScale + ReLU       │
    │    ↓                                                        │
    │  (B, 8, 23, 60, 60) ← 时间维度保持 23！                    │
    │    ↓                                                        │
    │  Layer 2: Conv3d(1,15,15) + BN + SpatialScale + ReLU       │
    │    ↓                                                        │
    │  (B, 8, 23, 46, 46) ← 时间维度仍为 23！                    │
    │    ↓                                                        │
    │  Permute + Flatten Spatial                                 │
    │    ↓                                                        │
    │  (B, 23, 8*46*46) = (B, 23, 16928)                         │
    │    ↓                                                        │
    │  Linear Readout: (B, 23, N_neurons)                        │
    │    ↓                                                        │
    │  + BehaviorModulator Offset                                │
    │    ↓                                                        │
    │  Softplus → Final Output: (B, 23, N_neurons) ≥ 0           │
    └─────────────────────────────────────────────────────────────┘

    关键特性:
    1. 时间维度全程保留 (23帧)
    2. Spatial Scaling Parameters 在 ReLU 前应用
    3. Softplus 输出保证 Firing Rate ≥ 0
    """

    def __init__(
        self,
        n_neurons: int = 50,
        n_channels: int = 8,
        input_shape: Tuple[int, int, int] = (23, 80, 80)
    ):
        super().__init__()

        self.n_neurons = n_neurons
        self.n_channels = n_channels
        self.time_frames, self.height, self.width = input_shape

        # ==========================================
        # Layer 1: Spatiotemporal Conv3d (保留时间维度!)
        # ==========================================
        # 时间核=5, 空间核=21×21
        # padding=(2, 0, 0) 使时间维度保持 23 (23 - 5 + 1 + 2*2 = 23)
        self.conv1 = nn.Conv3d(
            in_channels=1,
            out_channels=n_channels,
            kernel_size=(5, 21, 21),
            stride=1,
            padding=(2, 0, 0)  # 时间 padding=2，空间无 padding
        )
        self.bn1 = nn.BatchNorm3d(n_channels)

        # 输出空间尺寸: 80 - 21 + 1 = 60
        # 输出时间尺寸: 23 (保持不变!)
        self.layer1_spatial_out = self.height - 21 + 1  # 60

        # Spatial Scaling Parameter (5D for Conv3d output)
        # 形状: (1, 8, 1, 60, 60) - 利用广播机制
        self.spatial_scale_1 = nn.Parameter(
            torch.ones(1, n_channels, 1, self.layer1_spatial_out, self.layer1_spatial_out)
        )

        # ==========================================
        # Layer 2: Spatial Conv3d (时间核=1)
        # ==========================================
        # 时间核=1, 空间核=15×15
        # padding=0，时间维度不变 (23 - 1 + 1 = 23)
        self.conv2 = nn.Conv3d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=(1, 15, 15),
            stride=1,
            padding=0
        )
        self.bn2 = nn.BatchNorm3d(n_channels)

        # 输出空间尺寸: 60 - 15 + 1 = 46
        # 输出时间尺寸: 23 (保持不变!)
        self.layer2_spatial_out = self.layer1_spatial_out - 15 + 1  # 46

        # Spatial Scaling Parameter (5D for Conv3d output)
        # 形状: (1, 8, 1, 46, 46)
        self.spatial_scale_2 = nn.Parameter(
            torch.ones(1, n_channels, 1, self.layer2_spatial_out, self.layer2_spatial_out)
        )

        # ==========================================
        # Layer 3: Readout & Behavior Modulator
        # ==========================================
        # 展平空间维度，保留时间维度
        # (B, 8, 23, 46, 46) -> (B, 23, 8*46*46) = (B, 23, 16928)
        self.flatten_dim = n_channels * self.layer2_spatial_out ** 2  # 8 * 46 * 46 = 16928

        # 逐时间步 Linear 映射
        self.fc_readout = nn.Linear(self.flatten_dim, n_neurons)

        # 行为调制器
        self.behavior_modulator = BehaviorModulator(
            behavior_dim=2,
            n_neurons=n_neurons,
            hidden_dim=32
        )

    def forward(
        self,
        stimulus: torch.Tensor,
        behavior: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播 (时间维度始终为 23)

        Args:
            stimulus: (B, 1, 23, 80, 80) - 视觉刺激
            behavior: (B, 23, 2) or None - 行为时间序列

        Returns:
            firing_rate: (B, 23, N_neurons) - 逐帧放电率 (Softplus 保证 ≥ 0)
        """
        # Shape assertion
        assert stimulus.dim() == 5, f"Expected 5D tensor, got {stimulus.dim()}D"
        B = stimulus.shape[0]

        # ==========================================
        # Layer 1: Spatiotemporal Conv
        # ==========================================
        # Input: (B, 1, 23, 80, 80)
        x = self.conv1(stimulus)  # (B, 8, 23, 60, 60) ← 时间维度 23 保持不变!
        assert x.shape[2] == 23, f"Expected time dim = 23, got {x.shape[2]}"

        x = self.bn1(x)
        x = x * self.spatial_scale_1  # 空间缩放 (广播)
        x = F.relu(x)
        # Shape: (B, 8, 23, 60, 60)

        # ==========================================
        # Layer 2: Spatial Conv (时间核=1)
        # ==========================================
        x = self.conv2(x)  # (B, 8, 23, 46, 46) ← 时间维度仍为 23!
        assert x.shape[2] == 23, f"Expected time dim = 23 after Layer 2, got {x.shape[2]}"

        x = self.bn2(x)
        x = x * self.spatial_scale_2  # 空间缩放 (广播)
        x = F.relu(x)
        # Shape: (B, 8, 23, 46, 46)

        # ==========================================
        # Layer 3: Permute & Flatten + Readout
        # ==========================================
        # Permute: (B, C, T, H, W) -> (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # (B, 23, 8, 46, 46)

        # Flatten 空间维度
        x = x.reshape(B, 23, -1)  # (B, 23, 16928)

        # Linear Readout (逐时间步)
        base_response = self.fc_readout(x)  # (B, 23, N_neurons)

        # ==========================================
        # Behavior Modulation (逐帧)
        # ==========================================
        if behavior is not None:
            offset = self.behavior_modulator(behavior)  # (B, 23, N_neurons)
            output = base_response + offset
        else:
            output = base_response

        # ==========================================
        # Output: Firing Rate (non-negative)
        # ==========================================
        # Softplus: output >= 0 (Firing Rate)
        firing_rate = F.softplus(output)  # (B, 23, N_neurons) >= 0
        return firing_rate

    def get_spatial_scales(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取空间缩放参数（用于可视化和分析）"""
        return self.spatial_scale_1.data, self.spatial_scale_2.data

    def count_parameters(self) -> int:
        """计算可训练参数总数"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    n_neurons: int = 50,
    model_type: str = "DeepRetina3D",
    n_channels: int = 8,
    input_shape: Tuple[int, int, int] = (46, 80, 80),
    device: torch.device = None
) -> nn.Module:
    """
    工厂函数：创建模型实例

    Args:
        n_neurons: 神经元数量
        model_type: "DeepRetina3D" 或 "DeepRetinaSpatiotemporal"
        n_channels: 卷积通道数 (仅 DeepRetinaSpatiotemporal 使用)
        input_shape: (Time, Height, Width)
        device: 目标设备

    Returns:
        model: 模型实例
    """
    if model_type == "DeepRetina3D":
        model = DeepRetina3D(n_neurons=n_neurons)
    elif model_type == "DeepRetinaSpatiotemporal":
        model = DeepRetinaSpatiotemporal(
            n_neurons=n_neurons,
            n_channels=n_channels,
            input_shape=input_shape
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if device is not None:
        model = model.to(device)

    return model


# ==========================================
# Test Code
# ==========================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    n_neurons = 50
    batch_size = 4

    # ==========================================
    # Test 1: DeepRetina3D with FiLM
    # ==========================================
    print("\n" + "=" * 60)
    print("Testing DeepRetina3D + FiLM")
    print("=" * 60)

    model_3d = create_model(n_neurons=n_neurons, model_type="DeepRetina3D", device=device)

    print(f"\n[Architecture]")
    print(f"  Input:  Stimulus (B, 1, 33, 80, 80) + Behavior (B, 23, 2)")
    print(f"  Layer1: Conv3d(1->32, k=(5,7,7), p=(0,3,3)) -> (B, 32, 29, 40, 40)")
    print(f"  Layer2: Conv3d(32->64, k=(5,5,5), p=(0,2,2)) -> (B, 64, 25, 20, 20)")
    print(f"  Layer3: Conv3d(64->64, k=(3,5,5), p=(0,2,2)) -> (B, 64, 23, 10, 10) + Residual")
    print(f"  Gaussian Factorized Readout:")
    print(f"    - mu: (n_neurons, 2) = {n_neurons * 2:,} params [RF center]")
    print(f"    - sigma: (n_neurons, 1) = {n_neurons * 1:,} params [RF size]")
    print(f"    - feature_weights: (n_neurons, 64) = {n_neurons * 64:,} params")
    print(f"    - bias: (n_neurons,) = {n_neurons:,} params [init=-2.0]")
    print(f"    - Total: {n_neurons * 2 + n_neurons * 1 + n_neurons * 64 + n_neurons:,} params")
    print(f"  FiLM: gain/shift -> (64,) feature modulation")
    print(f"  Softplus -> Final: (B, 23, {n_neurons}) [Non-negative]")
    print(f"\nTotal parameters: {model_3d.count_parameters():,}")

    # Test inputs
    stimulus_33 = torch.randn(batch_size, 1, 33, 80, 80, device=device)
    behavior = torch.randn(batch_size, 23, 2, device=device)

    # ==========================================
    # Test 1a: Forward without behavior
    # ==========================================
    print(f"\n[Forward Pass - No Behavior]")
    output_no_beh = model_3d(stimulus_33, None)
    print(f"  Input: {stimulus_33.shape}")
    print(f"  Output: {output_no_beh.shape}")
    assert output_no_beh.shape == (batch_size, 23, n_neurons)
    assert (output_no_beh >= 0).all()
    print(f"  [OK] Shape correct, all outputs >= 0")

    # ==========================================
    # Test 1b: Forward with behavior (FiLM active)
    # ==========================================
    print(f"\n[Forward Pass - With FiLM Behavior]")
    output_with_beh = model_3d(stimulus_33, behavior)
    print(f"  Input: stimulus {stimulus_33.shape}, behavior {behavior.shape}")
    print(f"  Output: {output_with_beh.shape}")
    assert output_with_beh.shape == (batch_size, 23, n_neurons)
    assert (output_with_beh >= 0).all()
    print(f"  [OK] Shape correct, all outputs >= 0")

    # ==========================================
    # Test 1c: FiLM effect verification
    # ==========================================
    print(f"\n[FiLM Effect Verification]")
    # Outputs should differ when behavior is provided
    assert not torch.allclose(output_no_beh, output_with_beh), \
        "FiLM should modify the output!"
    print(f"  [OK] FiLM modifies output (no_beh vs with_beh differ)")

    # Verify gain >= 1.0
    gain_raw = model_3d.gain_net(behavior)
    gain = 1.0 + F.softplus(gain_raw)
    assert (gain >= 1.0).all(), "Gain must be >= 1.0"
    print(f"  [OK] Gain values >= 1.0 (min={gain.min().item():.4f}, max={gain.max().item():.4f})")

    # ==========================================
    # Test 1d: Gradient check
    # ==========================================
    print(f"\n[Gradient Check]")
    target = torch.rand(batch_size, 23, n_neurons, device=device) * 5
    criterion = nn.PoissonNLLLoss(log_input=False, full=True)
    loss = criterion(output_with_beh, target)
    loss.backward()

    print(f"  Loss (PoissonNLL): {loss.item():.4f}")
    print(f"  conv1.weight.grad: {model_3d.conv1.weight.grad is not None}")
    print(f"  conv2.weight.grad: {model_3d.conv2.weight.grad is not None}")
    print(f"  conv3.weight.grad: {model_3d.conv3.weight.grad is not None}")
    print(f"  mu.grad: {model_3d.mu.grad is not None}")
    print(f"  sigma.grad: {model_3d.sigma.grad is not None}")
    print(f"  feature_weights.grad: {model_3d.feature_weights.grad is not None}")
    print(f"  bias.grad: {model_3d.bias.grad is not None}")
    print(f"  gain_net.weight.grad: {model_3d.gain_net.weight.grad is not None}")
    print(f"  shift_net.weight.grad: {model_3d.shift_net.weight.grad is not None}")

    print(f"\n[OK] DeepRetina3D + Gaussian Factorized Readout tests passed!")

    # ==========================================
    # Test 2: DeepRetinaSpatiotemporal (Legacy)
    # ==========================================
    print("\n" + "=" * 60)
    print("Testing DeepRetinaSpatiotemporal (Seq2Seq)")
    print("=" * 60)

    model_s2s = create_model(
        n_neurons=n_neurons,
        model_type="DeepRetinaSpatiotemporal",
        input_shape=(23, 80, 80),
        device=device
    )

    print(f"\nTotal parameters: {model_s2s.count_parameters():,}")

    stimulus_23 = torch.randn(batch_size, 1, 23, 80, 80, device=device)
    behavior_23 = torch.randn(batch_size, 23, 2, device=device)

    output_s2s = model_s2s(stimulus_23, behavior_23)
    assert output_s2s.shape == (batch_size, 23, n_neurons)
    assert (output_s2s >= 0).all()

    print(f"  Input: {stimulus_23.shape}")
    print(f"  Output: {output_s2s.shape}")
    print(f"  [OK] DeepRetinaSpatiotemporal tests passed!")

    print("\n" + "=" * 60)
    print("[OK] All model tests passed!")
    print("=" * 60)
