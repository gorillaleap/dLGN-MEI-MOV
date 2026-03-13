import numpy as np
import h5py
from cascade2p import cascade
import matplotlib.pyplot as plt
import torch
import os

def process_offline():
    # --- 配置区 ---
    mat_path = 'training_data2.mat'
    save_path = 'cascade_spikes_23frames.npy'
    cascade_model_name = 'Global_EXC_15Hz_smoothing100ms'
    
    # 检查 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 检测到设备: {device} " + (f"({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "⚠️ 请注意：正在使用 CPU，速度较慢"))

    # 1. 加载数据
    print(f"\n1. 加载数据: {mat_path}")
    with h5py.File(mat_path, 'r') as f:
        raw_response = f['responses'][:].T  # (85, 1440, 50)
    
    time_frames, n_trials, n_neurons = raw_response.shape
    print(f"   原始形状: {raw_response.shape}")
    
    # 2. 数据预处理
    # 维度变换与拍平: (85, 1440, 50) -> (1440, 50, 85) -> (72000, 85)
    dff_flat = raw_response.transpose(1, 2, 0).reshape(-1, time_frames)
    
    # 镜像填充 -> (72000, 205)
    PAD_WIDTH = 60
    dff_padded = np.pad(dff_flat, pad_width=((0, 0), (PAD_WIDTH, PAD_WIDTH)), mode='reflect')
    
    # 3. GPU 加速推断
    print(f"\n2. 开始 GPU 加速推断 (总计 {dff_padded.shape[0]} 条序列)...")
    
    # 注意：如果你的 cascade 库是 torch 版且 predict 支持 GPU
    # 我们分批处理防止显存炸掉 (Batch Size = 10000)
    batch_size = 1000
    all_spikes = []
    
    # 确保模型在当前目录
    if not os.path.exists(os.path.join('.', cascade_model_name)):
        print(f"⚠️ 警告: 在当前目录未找到模型文件夹 {cascade_model_name}")

    for i in range(0, dff_padded.shape[0], batch_size):
        batch_data = dff_padded[i : i + batch_size]
        
        # 调用 predict。如果库内部写得好，它会自动检测到你安装了 GPU 版 Torch 并加速
        # 传入 model_folder='.' 确保它读你下载好的模型
        batch_spikes = cascade.predict(cascade_model_name, batch_data, model_folder='.')
        all_spikes.append(batch_spikes)
        print(f"   进度: {min(i + batch_size, dff_padded.shape[0])}/{dff_padded.shape[0]}")

        # 🚀 手动清理显存垃圾
        torch.cuda.empty_cache()

    spikes_padded_flat = np.concatenate(all_spikes, axis=0)

    # 4. 还原与截取 (23帧)
    print("\n3. 数据后处理与截取...")
    # 去掉填充 -> (72000, 85)
    spikes_85 = spikes_padded_flat[:, PAD_WIDTH:-PAD_WIDTH]
    # 截取刺激窗口 [32:55] -> (72000, 23)
    spikes_23 = spikes_85[:, 32:55]
    # 还原形状 -> (1440, 50, 23)
    spikes_3d = spikes_23.reshape(n_trials, n_neurons, 23)
    # 转置回输出格式 -> (23, 1440, 50)
    final_spikes = spikes_3d.transpose(2, 0, 1)

    # 5. 保存结果
    np.save(save_path, final_spikes)
    print(f"✅ 成功！最终形状: {final_spikes.shape}，保存至: {save_path}")

    # 6. 调用校验绘图
    verify_and_plot(raw_response, final_spikes)

def verify_and_plot(raw_response, final_spikes):
    print("\n4. 正在生成验证图 (随机挑选 5 个强响应神经元)...")
    
    # 计算平均活跃度，选最强的神经元
    mean_activity = np.mean(final_spikes, axis=(0, 1)) 
    top_neurons = np.argsort(mean_activity)[-5:]
    
    trial_idx = np.random.randint(0, final_spikes.shape[1]) # 随机选一个 Trial 看看
    
    plt.figure(figsize=(12, 12))
    for i, neuron_idx in enumerate(reversed(top_neurons)):
        plt.subplot(5, 1, i+1)
        
        # 原始 dF/F (蓝色)
        dff = raw_response[:, trial_idx, neuron_idx]
        dff_norm = (dff - dff.min()) / (dff.max() - dff.min() + 1e-6)
        
        # 预测 Spikes (红色)
        spikes = final_spikes[:, trial_idx, neuron_idx]
        
        plt.plot(range(85), dff_norm, color='royalblue', alpha=0.5, label='Original dF/F')
        # 对齐到 32-54 帧
        plt.bar(range(32, 55), spikes, color='crimson', alpha=0.7, width=0.5, label='Cascade Spikes')
        
        plt.title(f"Neuron #{neuron_idx} | Trial {trial_idx}")
        if i == 0: plt.legend()
        plt.ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig('spike_check.png')
    print("📈 校验图已保存为 'spike_check.png'，请打开查看。")
    plt.show()

if __name__ == "__main__":
    process_offline()