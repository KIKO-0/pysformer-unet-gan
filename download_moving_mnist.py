# 下载Moving MNIST数据集
import os
import gzip
import numpy as np
from PIL import Image
from urllib.request import urlretrieve

# 配置
DATA_URL = "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"
SAVE_DIR = "data/moving_mnist_radar"
SEQ_LEN = 20  # 每个序列包含20张图


def download_and_convert():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # 1. 下载原始 .npy 文件
    npy_path = "mnist_test_seq.npy"
    if not os.path.exists(npy_path):
        print(f"正在下载 Moving MNIST 数据集 (约 80MB)...")
        urlretrieve(DATA_URL, npy_path)
        print("下载完成！")

    # 2. 读取并转换
    print("正在解压并转换为图片序列，请稍候...")
    # 数据形状: (20, 10000, 64, 64) -> (序列长度, 样本数, 高, 宽)
    data = np.load(npy_path)

    # 我们只取前 200 个序列做实验（足够答辩用了，太多跑得慢）
    num_samples = 200

    # 将其展平为一个长序列文件夹，方便 dataset_precip.py 读取
    # 注意：为了让简单的 PrecipitationDirectory 能用，我们把所有序列拼接
    # 但为了逻辑正确，最好在文件名里标记序列ID

    cnt = 0
    for i in range(num_samples):
        seq_dir = os.path.join(SAVE_DIR, f"seq_{i:03d}")
        os.makedirs(seq_dir, exist_ok=True)
        for t in range(SEQ_LEN):
            img_array = data[t, i]
            img = Image.fromarray(img_array)
            # 保存为 seq_001/frame_001.png
            img.save(os.path.join(seq_dir, f"frame_{t:02d}.png"))
            cnt += 1

    print(f"处理完成！真实数据已保存在 {SAVE_DIR}")
    print(f"共生成 {num_samples} 个序列，合计 {cnt} 张图片。")


if __name__ == "__main__":
    download_and_convert()