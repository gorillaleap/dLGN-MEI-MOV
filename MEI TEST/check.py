import numpy as np
import matplotlib.pyplot as plt
import h5py
import random
import warnings
from cascade2p import cascade  # 确保这里用的是你本地能 work 的名字

def visualize_cascade_spikes(
    mat_path: str = 'training_data.mat',
    n_neurons_to_plot: int = 5,
    n_trials_to_plot: int = 5,
    model_name: str = 'Global_EXC_15Hz_smoothing100ms'
):
    print(f"正在从 {mat_path} 加载数据...")
    with h5py.File(mat_path, 'r') as f:
        raw_response = f['responses'][:].T  # (Time, Trials, Neurons)
        
    time_frames, n_total_trials, n_total_neurons = raw_response.shape
    print(f"原始响应数据维度: {raw_response.shape}")

    sel_trials = sorted(random.sample(range(n_total_trials), n_trials_to_plot))
    sel_neurons = sorted(random.sample(range(n_total_neurons), n_neurons_to_plot))

    try:
        cascade.download_model(model_name)
    except:
        pass 

    # ==========================================
    # 🚀 核心突破点：科学的时间序列拼接法
    # ==========================================
    # 1. 提取选中神经元的所有 Trial 数据: shape -> (Time, 1440, 5)
    neuron_data = raw_response[:, :, sel_neurons]
    
    # 2. 将时间轴和 Trial 轴拍扁拼接: 转置为 (1440, Time, 5)，然后展平为 (1440 * Time, 5)
    continuous_data = neuron_data.transpose(1, 0, 2).reshape(-1, n_neurons_to_plot)
    
    # 3. 转置为 CASCADE 期望的输入格式: (Neurons, Total_Time)
    continuous_dff = continuous_data.T
    
    print(f"✅ 已拼接为连续序列，长度: {continuous_dff.shape[1]} 帧，正在运行 CASCADE...")
    
    # 4. 一次性进行全局去卷积（完美解决 Noise 估计失败导致的 NaN）
    continuous_spikes = cascade.predict(model_name, continuous_dff)
    
    # 5. 把长序列重新切回原本的 Trial 形状: (1440, Time, 5)
    spikes_reshaped = continuous_spikes.T.reshape(n_total_trials, time_frames, n_neurons_to_plot)
    # ==========================================

    fig, axes = plt.subplots(n_neurons_to_plot, n_trials_to_plot, figsize=(20, 12))
    fig.suptitle(f"CASCADE2p Deconvolution: Raw dF/F vs Inferred Spikes\n(Random {n_trials_to_plot} Trials x {n_neurons_to_plot} Neurons)", fontsize=18, y=0.98)
    time_axis = np.arange(time_frames)

    for col, trial_idx in enumerate(sel_trials):
        for row, neuron_idx in enumerate(sel_neurons):
            ax_dff = axes[row, col]
            
            # 从重组好的矩阵中提取当前画图所需的数据
            dff_trace = neuron_data[:, trial_idx, row]
            spike_trace = spikes_reshaped[trial_idx, :, row]
            
            # --- 画原始 dF/F (蓝色线) ---
            color_dff = 'tab:blue'
            ax_dff.plot(time_axis, dff_trace, color=color_dff, linewidth=2, label='dF/F')
            ax_dff.tick_params(axis='y', labelcolor=color_dff)
            
            if row == n_neurons_to_plot - 1:
                ax_dff.set_xlabel('Time (frames)', fontsize=10)
            if col == 0:
                ax_dff.set_ylabel(f'Neuron {neuron_idx}\ndF/F', color=color_dff, fontsize=12, fontweight='bold')
            if row == 0:
                ax_dff.set_title(f'Trial {trial_idx}', fontsize=14, fontweight='bold')

            # --- 画 Spike (红色柱状图) ---
            ax_spike = ax_dff.twinx()
            color_spike = 'tab:red'
            
            # 🚀 终极防护罩：万一还有 NaN，强行归零防止崩溃
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                valid_max = np.nanmax(spike_trace) if not np.isnan(spike_trace).all() else 0.0
            spike_trace_clean = np.nan_to_num(spike_trace, nan=0.0)
            
            ax_spike.bar(time_axis, spike_trace_clean, color=color_spike, alpha=0.6, width=0.5, label='Spike Rate')
            ax_spike.tick_params(axis='y', labelcolor=color_spike)
            ax_spike.set_ylim(bottom=0, top=max(valid_max * 1.2, 0.01))
                
            ax_dff.grid(False)
            ax_spike.grid(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = 'cascade_sanity_check.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 绘图完成！图片已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    visualize_cascade_spikes()