"""
Calcium Imaging Dataset for Spike Prediction - Sequence-to-Sequence Format

全量预加载版本：所有数据在 __init__ 中一次性加载到内存

核心特性:
    - Stimulus 预裁切 RF + 预归一化 -> (1440, 1, 23, 80, 80) float32
    - Behavior 预转置 -> (1440, 23, 2) float32
    - Response 预转置 -> (1440, 23, 50) float32
    - __getitem__ 仅做内存切片，无任何计算，极速训练

内存估算:
    Stimulus: 1440 x 1 x 23 x 80 x 80 x 4 bytes ~ 4.0 GB
    Behavior: 1440 x 23 x 2 x 4 bytes ~ 0.3 GB
    Response: 1440 x 23 x 50 x 4 bytes ~ 6.6 GB
    Total: ~10.9 GB (modern workstation acceptable)

依赖安装:
    pip install cascade2p

Author: Refined Version
Date: 2026-03-10
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
import h5py
from tqdm import tqdm

# ==========================================
# CASCADE2p 导入 (可选依赖)
# ==========================================
CASCADE_AVAILABLE = False
try:
    from cascade2p import cascade
    CASCADE_AVAILABLE = True
except ImportError:
    pass  # 静默处理


class CalciumImagingDataset(Dataset):
    def __init__(
        self,
        stimulus: Union[str, np.ndarray, torch.Tensor],
        behavior: Optional[Union[str, np.ndarray, torch.Tensor]] = None,
        response: Optional[Union[str, np.ndarray, torch.Tensor]] = None,
        expand_factor: int = 30,
        rf_diameter: int = 80,
        rf_center: Optional[Tuple[int, int]] = None,
        is_training: bool = True,
        normalize: bool = True,
        jitter_range: int = 5,
        stim_mean: Optional[float] = None,
        stim_std: Optional[float] = None,
        use_cascade: bool = False,
        cascade_model: str = 'Global_EXC_15Hz_smoothing100ms',
        frame_rate: float = 15.5,
        model_folder: str = './cascade_models',
        **kwargs  # 兼容旧参数
    ):
        self.expand_factor = expand_factor
        self.rf_diameter = rf_diameter
        self.rf_radius = rf_diameter // 2
        self.is_training = is_training
        self.normalize = normalize
        self.jitter_range = jitter_range  # 保留但预加载版本不使用
        self.use_cascade = use_cascade and CASCADE_AVAILABLE
        self.cascade_model = cascade_model
        self.frame_rate = frame_rate
        self.model_folder = model_folder

        # 存储设备 (CPU for DataLoader pin_memory)
        self.storage_device = torch.device('cpu')

        # ==========================================
        # 1. 加载原始 Stimulus
        # ==========================================
        print("\n[CalciumImagingDataset] Pre-loading all data into memory...")
        raw_stimulus = self._load_stimulus(stimulus)

        # 获取维度
        self.n_base_trials, self.time_frames, self.full_height, self.full_width = raw_stimulus.shape
        self.n_samples = self.n_base_trials * self.expand_factor

        # ==========================================
        # 2. 计算 Stimulus 归一化参数
        # ==========================================
        if stim_mean is not None and stim_std is not None:
            self.stim_mean = stim_mean
            self.stim_std = stim_std
        elif normalize:
            stim_float = raw_stimulus.float()
            self.stim_mean = stim_float.mean().item()
            self.stim_std = stim_float.std().item()
            del stim_float
        else:
            self.stim_mean = 0.0
            self.stim_std = 1.0

        # ==========================================
        # 3. 感受野中心
        # ==========================================
        if rf_center is not None:
            self.rf_center_h, self.rf_center_w = rf_center
        else:
            self.rf_center_h = self.full_height // 2
            self.rf_center_w = self.full_width // 2

        # ==========================================
        # 4. 预处理所有 Stimulus (一次性)
        # ==========================================
        self.all_stimuli = self._preprocess_all_stimuli(
            raw_stimulus,
            self.stim_mean,
            self.stim_std
        ).float()

        # 释放原始 stimulus
        del raw_stimulus

        # ==========================================
        # 5. 加载并预处理 Behavior
        # ==========================================
        if behavior is not None:
            self.all_behaviors = self._load_behavior(behavior).float()
        else:
            self.all_behaviors = torch.zeros(self.n_samples, self.time_frames, 2)

        # ==========================================
        # 6. 加载并预处理 Response (极其严格)
        # ==========================================
        if response is not None:
            raw_response = self._load_response(response)

            if self.use_cascade:
                self.spikes = self._deconvolve_with_cascade(raw_response)
            else:
                self.spikes = raw_response

            # 转置为 (Trials, Time, N_neurons)
            self.all_responses = self.spikes.permute(1, 0, 2).float()
        else:
            self.all_responses = torch.zeros(self.n_samples, self.time_frames, 1)
            self.spikes = None

        # ==========================================
        # 7. 标准化 Behavior 和 Response
        # ==========================================
        if self.normalize:
            self._normalize_inplace()

        # 神经元数量
        self.n_neurons = self.all_responses.shape[2] if self.all_responses is not None else 1

        # ==========================================
        # 8. 形状强制检查 (最强防线)
        # ==========================================
        if self.all_responses.shape[1] != 23:
            raise ValueError(
                f"\n[FATAL ERROR] 响应数据维度错误!\n"
                f"  预期时间维为 23\n"
                f"  实际获得 shape: {self.all_responses.shape}\n"
                f"  请检查传入的 response 路径是否正确指向了 23 帧的 .npy 文件。\n"
            )
        print(f"[Dataset] Final response shape: {self.all_responses.shape} (Time dim = {self.all_responses.shape[1]})")

        # 打印信息
        self._print_info()

    def _load_stimulus(self, data) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            return data.to(self.storage_device)
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(self.storage_device)
        elif isinstance(data, str):
            path = Path(data)
            if path.suffix == '.npy':
                arr = np.load(path)
                return torch.from_numpy(arr).to(self.storage_device)
            elif path.suffix in ['.h5', '.hdf5', '.mat']:
                with h5py.File(path, 'r') as f:
                    keys_to_try = ['stimulus', 'final_matrix', 'data']
                    found_key = None
                    for key in keys_to_try:
                        if key in f:
                            found_key = key
                            break
                    if found_key is None:
                        raise KeyError(f"Cannot find stimulus in {path}. Keys: {list(f.keys())}")
                    arr = f[found_key][:]
                    if arr.ndim == 4:
                        arr = arr.T  # h5py 倒序修正
                    print(f"  [Loaded] stimulus from '{found_key}': {arr.shape}, dtype={arr.dtype}")
                    return torch.from_numpy(arr).to(self.storage_device)
            else:
                raise ValueError(f"Unsupported format: {path.suffix}")
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def _load_behavior(self, data) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            tensor = data.float().to(self.storage_device)
        elif isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data.astype(np.float32)).to(self.storage_device)
        elif isinstance(data, str):
            with h5py.File(data, 'r') as f:
                keys_to_try = ['behavior', 'Behaviour', 'data']
                for key in keys_to_try:
                    if key in f:
                        arr = f[key][:].T
                        tensor = torch.from_numpy(arr.astype(np.float32)).to(self.storage_device)
                        break
                else:
                    raise KeyError(f"Cannot find behavior in {data}")
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        if tensor.dim() == 3:
            tensor = tensor.permute(2, 1, 0)
        return tensor

    def _load_response(self, data) -> torch.Tensor:
        """加载 Response - 仅允许 23 帧"""
        if isinstance(data, torch.Tensor):
            tensor = data.float().to(self.storage_device)
            if tensor.shape[0] != 23:
                raise ValueError(f"[FATAL] Tensor 帧数不是 23! Got shape: {tensor.shape}.")
            return tensor

        elif isinstance(data, np.ndarray):
            if data.shape[0] != 23:
                raise ValueError(f"[FATAL] ndarray 帧数不是 23! Got shape: {data.shape}.")
            return torch.from_numpy(data.astype(np.float32)).to(self.storage_device)

        elif isinstance(data, str):
            path = Path(data)
            print(f"  [Response Loader] 尝试读取: {path.name}")
            
            if path.suffix == '.npy':
                if not path.exists():
                    raise FileNotFoundError(f"[FATAL] 找不到 .npy 文件: {path}")
                arr = np.load(path)
                print(f"  [Loaded] response from .npy: {arr.shape}, dtype={arr.dtype}")
                if arr.shape[0] != 23:
                    raise ValueError(f"[FATAL] 加载了 .npy 文件但帧数不是 23! Got shape: {arr.shape}.")
                return torch.from_numpy(arr.astype(np.float32)).to(self.storage_device)

            elif path.suffix in ['.h5', '.hdf5', '.mat']:
                with h5py.File(path, 'r') as f:
                    keys_to_try = ['responses', 'response', 'data']
                    for key in keys_to_try:
                        if key in f:
                            arr = f[key][:].T
                            break
                    else:
                        raise KeyError(f"Cannot find response in {data}")
                
                if not self.use_cascade and arr.shape[0] != 23:
                    raise ValueError(f"[FATAL] 读取 .mat 文件响应，预期 23 帧但获得 {arr.shape}。请直接传入预处理好的 .npy 文件路径！")
                return torch.from_numpy(arr.astype(np.float32)).to(self.storage_device)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def _deconvolve_with_cascade(self, raw_response: torch.Tensor) -> torch.Tensor:
        # (已停用) 略去内部逻辑以节省空间，维持原有结构
        raise RuntimeError("CASCADE 在线推断已禁用！请使用预处理的 .npy 文件并设置 use_cascade=False")

    def _preprocess_all_stimuli(self, raw_stim: torch.Tensor, mean: float, std: float) -> torch.Tensor:
        n_base, time_frames, H, W = raw_stim.shape
        n_total = n_base * self.expand_factor

        print(f"\n[_preprocess_all_stimuli] Preprocessing {n_total} samples...")
        all_stimuli = torch.zeros(
            n_total, 1, time_frames, self.rf_diameter, self.rf_diameter,
            dtype=torch.float32, device=self.storage_device
        )

        center_h = self.rf_center_h
        center_w = self.rf_center_w

        for i in tqdm(range(n_total), desc="Preprocessing stimuli"):
            base_idx = i // self.expand_factor
            stim = raw_stim[base_idx].float()

            start_h = center_h - self.rf_radius
            start_w = center_w - self.rf_radius
            end_h = center_h + self.rf_radius
            end_w = center_w + self.rf_radius

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
                stim_crop = torch.nn.functional.pad(
                    stim_crop, (pad_w_before, pad_w_after, pad_h_before, pad_h_after),
                    mode='constant', value=0
                )

            stim_crop = (stim_crop - mean) / (std + 1e-8)
            all_stimuli[i, 0] = stim_crop

        print(f"  [Done] Memory usage: {all_stimuli.element_size() * all_stimuli.numel() / 1024**3:.2f} GB")
        return all_stimuli

    def _normalize_inplace(self):
        if self.all_behaviors is not None:
            beh_mean = self.all_behaviors.mean(dim=(0, 1), keepdim=True)
            beh_std = self.all_behaviors.std(dim=(0, 1), keepdim=True)
            self.all_behaviors = (self.all_behaviors - beh_mean) / (beh_std + 1e-8)

        if self.all_responses is not None:
            self.all_responses = torch.clamp(self.all_responses, min=0)

    def _print_info(self):
        stim_mem = self.all_stimuli.element_size() * self.all_stimuli.numel() / 1024**3
        beh_mem = self.all_behaviors.element_size() * self.all_behaviors.numel() / 1024**3 if self.all_behaviors is not None else 0
        resp_mem = self.all_responses.element_size() * self.all_responses.numel() / 1024**3 if self.all_responses is not None else 0
        total_mem = stim_mem + beh_mem + resp_mem

        print(f"\n[CalciumImagingDataset] Pre-loaded (All-in-Memory)")
        print(f"  Total samples: {self.n_samples}")
        print(f"  Time frames: {self.time_frames}")
        print(f"  RF diameter: {self.rf_diameter}")
        print(f"  N_neurons: {self.n_neurons}")
        print(f"  CASCADE: {'Enabled' if self.use_cascade else 'Disabled'}")
        print(f"  Memory Usage: {total_mem:.3f} GB")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.all_stimuli[idx],      # (1, 23, 80, 80)
            self.all_behaviors[idx],    # (23, 2)
            self.all_responses[idx]     # (23, N_neurons)
        )
    
class SpikeDataset(Dataset):
    """
    模拟 Spike 数据集（用于本地测试管线，无真实数据依赖）
    生成符合 Sequence-to-Sequence 格式的随机张量
    """
    def __init__(
        self,
        n_trials: int = 1000,
        n_neurons: int = 50,
        time_frames: int = 23,
        rf_diameter: int = 80
    ):
        self.n_trials = n_trials
        self.n_neurons = n_neurons
        self.time_frames = time_frames
        self.rf_diameter = rf_diameter

    def __len__(self) -> int:
        return self.n_trials

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            stimulus: (1, 23, 80, 80)
            behavior: (23, 2)
            response: (23, 50) - 模拟的非负 Spike
        """
        stimulus = torch.randn(1, self.time_frames, self.rf_diameter, self.rf_diameter)
        behavior = torch.randn(self.time_frames, 2)
        response = torch.relu(torch.randn(self.time_frames, self.n_neurons)) + 0.1
        return stimulus, behavior, response

def create_dataloaders(
    dataset: Dataset,
    train_ratio: float = 0.8,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器
    使用固定种子的随机打乱划分，保证每次实验可复现。
    """
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)

    # 使用固定种子的随机打乱
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=generator).tolist()

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    print(f"\n[create_dataloaders] Random split with seed={seed}")
    print(f"  Train samples: {len(train_indices)} ({len(train_indices)/n_total*100:.1f}%)")
    print(f"  Val samples: {len(val_indices)} ({len(val_indices)/n_total*100:.1f}%)")

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练集打乱
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, # 验证集不需要打乱
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader

# ==========================================
# 核心改动：双路读取逻辑
# ==========================================
def load_training_dataset(
    mat_path: str,
    response_path: Optional[str] = None,  # 🚀 新增参数
    expand_factor: int = 30,
    rf_diameter: int = 80,
    rf_center: Optional[Tuple[int, int]] = None,
    is_training: bool = True,
    normalize: bool = True,
    use_cascade: bool = False,
    **kwargs
) -> CalciumImagingDataset:
    
    print(f"\n[load_training_dataset] Base Mat Path: {mat_path}")
    
    with h5py.File(mat_path, 'r') as f:
        # 1. 仅加载 Stimulus
        stim_raw = f['stimulus'][:].T
        stim_tensor = torch.from_numpy(stim_raw)

        # 2. 仅加载 Behavior
        beh_raw = f['behavior'][:].T
        beh_tensor = torch.from_numpy(beh_raw.astype(np.float32)).permute(0, 2, 1)

        # 3. 决定 Response 的来源 (走 .mat 还是 .npy)
        if response_path is None:
            print("  ⚠️ 未提供 response_path，将回退读取 .mat 文件内部的响应...")
            resp_raw = f['responses'][:].T
            response_data = torch.from_numpy(resp_raw.astype(np.float32))
        else:
            print(f"  ✅ 发现外部 response_path，准备加载: {response_path}")
            response_data = response_path  # 把字符串路径传进去，Dataset内部去接管

    print(f"\n[Creating Dataset with pre-loading]")
    dataset = CalciumImagingDataset(
        stimulus=stim_tensor,
        behavior=beh_tensor,
        response=response_data,  # 🚀 直接传入 .npy 的字符串路径
        expand_factor=expand_factor,
        rf_diameter=rf_diameter,
        rf_center=rf_center,
        is_training=is_training,
        normalize=normalize,
        use_cascade=use_cascade,
        **kwargs
    )

    return dataset