import numpy as np
import torch

def is_kspace(data):
    """
    检查输入数据是否是 k-space 数据。
    :param data: 输入数据（PyTorch 张量，可能在 GPU 上）。
    :return: True 如果是 k-space 数据，否则 False。
    """
    # 如果数据在 GPU 上，将其移动到 CPU 并转换为 NumPy 数组
    if isinstance(data, torch.Tensor):
        if data.is_cuda:
            data = data.cpu()  # 将数据从 GPU 移动到 CPU
        data = data.detach().numpy()  # 分离计算图并转换为 NumPy 数组
    # Input data magnitude - min: 9.313226e-10
    # Input data magnitude - max: 0.08225857
    # Input data magnitude - mean: 0.0018605998
    print("Input data magnitude - min:", np.min(np.abs(data)))
    print("Input data magnitude - max:", np.max(np.abs(data)))
    print("Input data magnitude - mean:", np.mean(np.abs(data)))
    # 检查数据的对称性或动态范围
    image = np.fft.ifft2(data)  # 对 k-space 数据进行逆傅里叶变换
    # is_kspace-np.max(np.abs(image)) 0.0012771301046804883
    print('is_kspace-np.max(np.abs(image))',np.max(np.abs(image)))
    # Image magnitude - min: 4.340007751680056e-11
    # Image magnitude - max: 0.0013321273498925043
    # Image magnitude - mean: 0.00015554498381737105
    print("Image magnitude - min:", np.min(np.abs(image)))
    print("Image magnitude - max:", np.max(np.abs(image)))
    print("Image magnitude - mean:", np.mean(np.abs(image)))
    return np.max(np.abs(image)) < 1e-3  # 假设图像域数据的幅度较小
# 检查输入数据是否满足共轭对称性,
def is_conjugate_symmetric(data):
    # 如果数据在 GPU 上，将其移动到 CPU 并转换为 NumPy 数组
    if isinstance(data, torch.Tensor):
        if data.is_cuda:
            data = data.cpu()  # 将数据从 GPU 移动到 CPU
        data = data.detach().numpy()  # 分离计算图并转换为 NumPy 数组
    data_conj = np.conj(data[::-1, ::-1])  # 共轭对称
    return np.allclose(data, data_conj, atol=1e-5)