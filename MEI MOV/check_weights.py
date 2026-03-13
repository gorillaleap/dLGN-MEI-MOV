import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置你的路径
PROJECT_ROOT = Path(__file__).parent
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pth"

def main():
    if not CHECKPOINT_PATH.exists():
        print(f"找不到权重文件: {CHECKPOINT_PATH}")
        return

    # 1. 强行拆包加载权重文件 (只加载字典，不实例化模型)
    print("Loading checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    
    # 兼容不同的保存格式
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # 2. 揪出那个嫌疑最大的读出层权重
    weight_key = 'feature_weights'
    if weight_key not in state_dict:
        print(f"没找到 {weight_key}！模型结构字典包含以下键:")
        for k in state_dict.keys():
            if 'readout' in k or 'weight' in k:
                print(f"  - {k}")
        return

    weights = state_dict[weight_key] # 形状应该是 (50, 576) 或类似
    print(f"\n成功提取特征权重，形状为: {weights.shape}")

    # 3. 只取前 5 个神经元做体检
    target_weights = weights[5:20].numpy()

    # 4. 计算它们两两之间的相似度 (相关系数矩阵)
    # np.corrcoef 会计算行与行之间的相关性
    corr_matrix = np.corrcoef(target_weights)

    print("\n" + "="*50)
    print("🧠 神经元 0-4 的权重相似度矩阵 (Correlation Matrix):")
    print("="*50)
    
    # 打印格式化的矩阵
    header = "       " + "".join([f"N{i:4d} " for i in range(5)])
    print(header)
    for i in range(5):
        row_str = f"N{i:<2d} | "
        for j in range(5):
            val = corr_matrix[i, j]
            # 用颜色标记严重程度 (终端可能不支持颜色，我们用符号代替)
            if i != j and val > 0.8:
                row_str += f"{val:5.2f}* " # 标红/打星号
            else:
                row_str += f"{val:5.2f}  "
        print(row_str)
    
    # 计算平均非对角线相关性
    off_diag_sum = np.sum(corr_matrix) - 5 # 减去对角线的 5 个 1
    mean_off_diag = off_diag_sum / (5 * 4)
    
    print("\n" + "="*50)
    print(f"📊 平均交叉相似度: {mean_off_diag:.4f}")
    
    if mean_off_diag > 0.9:
        print("🚨 终极警报: 极其严重的特征坍缩！它们就是克隆人！")
    elif mean_off_diag > 0.6:
        print("⚠️ 严重警告: 权重高度相似，模型趋向于提取单一主导特征。")
    else:
        print("✅ 健康: 神经元保持了良好的多样性。")
    print("="*50)

    # 画一张直观的热力图保存下来
    plt.figure(figsize=(6, 5))
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.title('Readout Weight Correlation (Neurons 0-4)')
    plt.xticks(ticks=np.arange(5), labels=[f"N {i}" for i in range(5)])
    plt.yticks(ticks=np.arange(5), labels=[f"N {i}" for i in range(5)])
    for i in range(5):
        for j in range(5):
            plt.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center", color="black" if abs(corr_matrix[i, j]) < 0.5 else "white")
    plt.tight_layout()
    save_path = PROJECT_ROOT / "weight_correlation.png"
    plt.savefig(save_path, dpi=150)
    print(f"\n[Saved] 相似度热力图已保存至: {save_path}")

if __name__ == "__main__":
    main()