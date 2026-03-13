import numpy as np
import os

file_path = 'cascade_spikes_23frames.npy'

if os.path.exists(file_path):
    # mmap_mode='r' 可以让你在不完全加载大文件到内存的情况下读取它的信息
    data = np.load(file_path, mmap_mode='r')
    
    print("-" * 40)
    print(f"📂 文件路径: {os.path.abspath(file_path)}")
    print(f"📐 张量维度 (Shape): {data.shape}")
    print(f"🔢 元素总数 (Size): {data.size}")
    print(f"🧪 数据类型 (Dtype): {data.dtype}")
    
    # 计算文件大小 (MB)
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"💾 磁盘占用: {file_size_mb:.2f} MB")
    print("-" * 40)

    # 逻辑判断提示
    if data.shape[0] == 23:
        print("✅ 确认：时间轴长度为 23，数据已经是预切片好的。")
    elif data.shape[0] == 85:
        print("❌ 警告：时间轴长度仍为 85，说明离线脚本没有执行截取！")
    else:
        print(f"❓ 提示：检测到非预期维度 {data.shape[0]}。")
else:
    print(f"❌ 错误：在当前目录下找不到 '{file_path}'。")
    print(f"当前目录下的文件有: {os.listdir('.')}")