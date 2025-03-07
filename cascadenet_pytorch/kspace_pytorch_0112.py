import numpy as np
import torch
import torch.nn as nn


import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import savemat
import torch

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import savemat
import torch

# def kspace_to_image(k_space):
#     """
#     将 k-space 数据转换到图像域。
    
#     参数:
#         k_space (np.ndarray): k-space 数据，形状为 [batch, channels, height, width, time]。
    
#     返回:
#         image (np.ndarray): 图像域数据，形状与输入相同。
#     """
#     # 在最后两个维度（height 和 width）进行逆傅里叶变换
#     image = np.fft.ifft2(k_space, axes=(-2, -1))
#     # 取幅值（可选，也可以取实部或虚部）
#     image = np.abs(image)
#     return image

# def save_data(data, save_dir, prefix, data_name):
#     """
#     将数据保存为图像和 .npy 文件。

#     参数:
#         data (np.ndarray): 需要保存的数据。
#         save_dir (str): 保存文件的目录。
#         prefix (str): 文件名前缀。
#         data_name (str): 数据名称（如 'k', 'k0', 'mask', 'out'）。
#     """
#     # 确保保存目录存在
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     # 保存为 .npy 文件
#     np.save(os.path.join(save_dir, f'{prefix}{data_name}.npy'), data)

#     # 保存为 .mat 文件
#     savemat(os.path.join(save_dir, f'{prefix}{data_name}.mat'), {data_name: data})

#     # 保存为图片
#     plt.imsave(os.path.join(save_dir, f'{prefix}{data_name}.png'), data[0, 0, :, :], cmap='gray')

#     print(f"Saved {data_name} data to {save_dir} with prefix '{prefix}'")
# def save_data(data, save_dir, prefix, data_name):
#     """
#     将数据保存为图像和 .npy 文件。

#     参数:
#         data (np.ndarray): 需要保存的数据。
#         save_dir (str): 保存文件的目录。
#         prefix (str): 文件名前缀。
#         data_name (str): 数据名称（如 'k', 'k0', 'mask', 'out'）。
#     """
#     # 确保保存目录存在
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     # 保存为 .npy 文件
#     np.save(os.path.join(save_dir, f'{prefix}{data_name}.npy'), data)

#     # 保存为 .mat 文件
#     savemat(os.path.join(save_dir, f'{prefix}{data_name}.mat'), {data_name: data})

#     # 确保数据是 2D 灰度图像
#     if data.ndim == 5:  # 如果数据是 5D 张量 [batch, channel, height, width, time]
#         image_data = data[0, 0, :, :, 0]  # 取第一个样本、第一个通道、第一个时间帧
#     elif data.ndim == 4:  # 如果数据是 4D 张量 [batch, channel, height, width]
#         image_data = data[0, 0, :, :]  # 取第一个样本、第一个通道
#     elif data.ndim == 3:  # 如果数据是 3D 张量 [channel, height, width]
#         image_data = data[0, :, :]  # 取第一个通道
#     else:  # 如果数据是 2D 张量 [height, width]
#         image_data = data

#     # 确保数据是实数（取幅值）
#     if np.iscomplexobj(image_data):
#         image_data = np.abs(image_data)

#     # 归一化到 [0, 1] 范围
#     image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))

#     # 保存为图片
#     plt.imsave(os.path.join(save_dir, f'{prefix}{data_name}.png'), image_data, cmap='gray')

#     print(f"Saved {data_name} data to {save_dir} with prefix '{prefix}'")

# def save_k_space_data(k, k0, mask, save_dir, prefix=''):
#     """
#     将 k-space 数据 k, k0 和 mask 保存为图片和 .npy 文件。

#     参数:
#         k (torch.Tensor): 输入的 k-space 数据。
#         k0 (torch.Tensor): 初始采样的 k-space 数据。
#         mask (torch.Tensor): 采样掩码。
#         save_dir (str): 保存文件的目录。
#         prefix (str): 文件名前缀。
#     """
#     # 将 Tensor 转换为 numpy 数组，并分离梯度
#     k_np = k.detach().cpu().numpy()
#     k0_np = k0.detach().cpu().numpy()
#     mask_np = mask.detach().cpu().numpy()

#     # 对 k 和 k0 进行逆傅里叶变换，转换到图像域
#     k_image = kspace_to_image(k_np)
#     k0_image = kspace_to_image(k0_np)

#     # 保存 k, k0, mask
#     save_data(k_image, save_dir, prefix, 'k_image')
#     save_data(k0_image, save_dir, prefix, 'k0_image')
#     save_data(mask_np, save_dir, prefix, 'mask')

# def save_out_data(out, save_dir, prefix=''):
#     """
#     将 out 保存为图像和 .npy 文件。

#     参数:
#         out (torch.Tensor): data_consistency 返回的 k-space 数据。
#         save_dir (str): 保存文件的目录。
#         prefix (str): 文件名前缀。
#     """
#     # 将 Tensor 转换为 numpy 数组，并分离梯度
#     out_np = out.detach().cpu().numpy()

#     # 将 k-space 数据转换到图像域
#     out_image = kspace_to_image(out_np)

#     # 保存 out
#     save_data(out_image, save_dir, prefix, 'out_image')
def save_k_space_data(k, k0, mask, save_dir, prefix=''):
    """
    将 k-space 数据 k, k0 和 mask 保存为图片和 .npy 文件。

    参数:
        k (torch.Tensor): 输入的 k-space 数据。
        k0 (torch.Tensor): 初始采样的 k-space 数据。
        mask (torch.Tensor): 采样掩码。
        save_dir (str): 保存文件的目录。
        prefix (str): 文件名前缀。
    """
    # 将 Tensor 转换为 numpy 数组，并分离梯度
    k_bak = k
    k_np = k_bak.detach().cpu().numpy()
    k0_bak =k0
    k0_np = k0_bak.detach().cpu().numpy()
    mask_bak = mask
    mask_np = mask_bak.detach().cpu().numpy()

    # 保存 k, k0, mask 为 .npy 和 .mat 格式
    np.save(os.path.join(save_dir, f'{prefix}k.npy'), k_np)
    savemat(os.path.join(save_dir, f'{prefix}k.mat'), {'k': k_np})

    np.save(os.path.join(save_dir, f'{prefix}k0.npy'), k0_np)
    savemat(os.path.join(save_dir, f'{prefix}k0.mat'), {'k0': k0_np})

    np.save(os.path.join(save_dir, f'{prefix}mask.npy'), mask_np)
    savemat(os.path.join(save_dir, f'{prefix}mask.mat'), {'mask': mask_np})
    
    # 将 k-space 数据转换到图像域
    k_image = kspace_to_image(k_np)
    k0_image = kspace_to_image(k0_np)

    # 保存 k, k0, mask
    save_data(k_image, save_dir, prefix, 'k_image')
    save_data(k0_image, save_dir, prefix, 'k0_image')

    print(f"Saved k-space data to {save_dir} with prefix '{prefix}'")
    
def save_out_data(out, save_dir, prefix=''):
    """
    将 out 保存为图像和 .npy 文件。

    参数:
        out (torch.Tensor): data_consistency 返回的 k-space 数据。
        save_dir (str): 保存文件的目录。
        prefix (str): 文件名前缀。
    """
    # 将 Tensor 转换为 numpy 数组，并分离梯度
    out_bak = out
    out_np = out_bak.detach().cpu().numpy()

    # 保存 out 为 .npy 和 .mat 格式
    np.save(os.path.join(save_dir, f'{prefix}out.npy'), out_np)
    savemat(os.path.join(save_dir, f'{prefix}out.mat'), {'out': out_np})
    
    # 将 k-space 数据转换到图像域
    out_image = kspace_to_image(out_np)
    # 保存 out为图像
    save_data(out_image, save_dir, prefix, 'out_image')

    print(f"Saved out data to {save_dir} with prefix '{prefix}'")

def kspace_to_image(k_space):
    """
    将 k-space 数据转换到图像域。
    
    参数:
        k_space (np.ndarray): k-space 数据，形状为 [batch, channels, height, width, time]。
    
    返回:
        image (np.ndarray): 图像域数据，形状与输入相同。
    """
    # 在最后两个维度（height 和 width）进行逆傅里叶变换
    #  # 逆傅里叶变换
    # x_res = torch.fft.ifft2(out, dim=(-2, -1), norm='backward')
    image = np.fft.ifft2(k_space, axes=(-2, -1))
    # 取幅值（可选，也可以取实部或虚部）
    image = np.abs(image)
    return image

def save_data(data, save_dir, prefix, data_name):
    """
    将数据保存为图像和 .npy 文件。

    参数:
        data (np.ndarray): 需要保存的数据。
        save_dir (str): 保存文件的目录。
        prefix (str): 文件名前缀。
        data_name (str): 数据名称（如 'k', 'k0', 'mask', 'out'）。
    """
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存为 .npy 文件
    np.save(os.path.join(save_dir, f'{prefix}{data_name}.npy'), data)

    # 保存为 .mat 文件
    savemat(os.path.join(save_dir, f'{prefix}{data_name}.mat'), {data_name: data})

    # 确保数据是 2D 灰度图像
    if data.ndim == 5:  # 如果数据是 5D 张量 [batch, channel, height, width, time]
        image_data = data[0, 0, :, :, 0]  # 取第一个样本、第一个通道、第一个时间帧
    elif data.ndim == 4:  # 如果数据是 4D 张量 [batch, channel, height, width]
        image_data = data[0, 0, :, :]  # 取第一个样本、第一个通道
    elif data.ndim == 3:  # 如果数据是 3D 张量 [channel, height, width]
        image_data = data[0, :, :]  # 取第一个通道
    else:  # 如果数据是 2D 张量 [height, width]
        image_data = data

    # 确保数据是实数（取幅值）
    if np.iscomplexobj(image_data):
        image_data = np.abs(image_data)

    # 归一化到 [0, 1] 范围
    image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))

    # 保存为图片
    plt.imsave(os.path.join(save_dir, f'{prefix}{data_name}.png'), image_data, cmap='gray')

    print(f"Saved {data_name} data to {save_dir} with prefix '{prefix}'")
  


# def check_data_domain(k, k0):
#     if is_kspace(k) and is_kspace(k0):
#         print("Both k and k0 are in k-space domain.")
#     elif not is_kspace(k) and not is_kspace(k0):
#         print("Both k and k0 are in image domain.")
#     else:
#         print("k and k0 are in different domains.")
           
# k 是当前 k 空间数据，k0 是原始采样数据，mask 是采样掩码。
def data_consistency(k, k0, mask, noise_lvl=None, save_dir=None, prefix='', save_last=False):
    """
    k    - input in k-space
    k0   - initially sampled elements in k-space
    mask - corresponding nonzero location
    noise_lvl - noise level
    save_dir - directory to save data (optional)
    prefix - prefix for saved files (optional)
    save_last - whether to save data (only for the last batch of the last epoch)
    """
    # Is input data conjugate symmetric-k: False
    # Is input data conjugate symmetric-k0: False
    # print("Is input data conjugate symmetric-k:", is_conjugate_symmetric(k))
    # print("Is input data conjugate symmetric-k0:", is_conjugate_symmetric(k0))
    # 检查k、k0是否在k空间
    # if not is_kspace(k) or not is_kspace(k0):
    #     raise ValueError("Input k and k0 must be in k-space domain.")
    
    # 原始逻辑
    v = noise_lvl
    if v:  # noisy case
        out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
    else:  # noiseless case
        out = (1 - mask) * k + mask * k0
    
    # if v:  # noisy case
    #     out = mask * (k + v * k0) / (1 + v)
    # else:  # noiseless case
    #     out =mask * k0

    # 如果需要保存数据（仅在最后一个 batch 的最后一个 epoch 保存）
    if save_last and save_dir:
        # save_k_space_data(k, k0, mask, save_dir, prefix)
        # save_out_data(out, save_dir, prefix)
         # 仅在测试阶段保存数据，避免中断计算图
        with torch.no_grad():
            save_k_space_data(k, k0, mask, save_dir, prefix)
            save_out_data(out, save_dir, prefix)
    return out

# def data_consistency(k, k0, mask, noise_lvl=None, save_dir=None, prefix=''):
#     """
#     k    - input in k-space
#     k0   - initially sampled elements in k-space
#     mask - corresponding nonzero location
#     noise_lvl - noise level
#     save_dir - directory to save data (optional)
#     prefix - prefix for saved files (optional)
#     """
#     # 如果需要保存输入数据
#     if save_dir:
#         save_k_space_data(k, k0, mask, save_dir, prefix)

#     # 原始逻辑
#     v = noise_lvl
#     if v:  # noisy case
#         out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
#     else:  # noiseless case
#         out = (1 - mask) * k + mask * k0

#     # 如果需要保存输出数据
#     if save_dir:
#         save_out_data(out, save_dir, prefix)
#     return out

# def data_consistency(k, k0, mask, noise_lvl=None):
#     """
#     k    - input in k-space
#     k0   - initially sampled elements in k-space
#     mask - corresponding nonzero location
#     """
#     v = noise_lvl
#     if v:  # noisy case
#         out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
#     else:  # noiseless case
#         out = (1 - mask) * k + mask * k0
#     return out



class DataConsistencyInKspace(nn.Module):
    def __init__(self, noise_lvl=None, norm='ortho'):
        super(DataConsistencyInKspace, self).__init__()
        self.normalized = norm == 'ortho'
        self.noise_lvl = noise_lvl

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def perform(self, x, k0, mask, save_last=False):
        """
        x    - input in image domain, of shape (n, 2, nx, ny[, nt])
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        save_last - whether to save data (only for the last batch of the last epoch)
        """
        if x.dim() == 4:  # input is 2D
            x = x.permute(0, 2, 3, 1)
            k0 = k0.permute(0, 2, 3, 1)
            mask = mask.permute(0, 2, 3, 1)
        elif x.dim() == 5:  # input is 3D
            x = x.permute(0, 4, 2, 3, 1)
            k0 = k0.permute(0, 4, 2, 3, 1)
            mask = mask.permute(0, 4, 2, 3, 1)

        # 傅里叶变换
        k = torch.fft.fft2(x, dim=(-2, -1), norm='forward')
        # 数据一致性处理
        out = data_consistency(k, k0, mask, self.noise_lvl, save_dir='./saved_data', prefix='last_epoch_', save_last=save_last)
        # k空间的最大值一般都是10点若干次方
        print("data_consistency out - min:", np.min(np.abs(out)))
        print("data_consistency out - max:",np.max(np.abs(out)))
        print("data_consistency out - mean:", np.mean(np.abs(out)))

        # 逆傅里叶变换
        x_res = torch.fft.ifft2(out, dim=(-2, -1), norm='backward')
        print("data_consistency x_res - min:", np.min(np.abs(x_res)))
        print("data_consistency x_res - max:",np.max(np.abs(x_res)))
        print("data_consistency x_res - mean:", np.mean(np.abs(x_res)))

        if x.dim() == 4:
            x_res = x_res.permute(0, 3, 1, 2)
        elif x.dim() == 5:
            x_res = x_res.permute(0, 4, 2, 3, 1)

        # 取幅值
        x_res = torch.abs(x_res)
        return x_res
# class DataConsistencyInKspace(nn.Module):
#     """ Create data consistency operator

#     Warning: note that FFT2 (by the default of torch.fft) is applied to the last 2 axes of the input.
#     This method detects if the input tensor is 4-dim (2D data) or 5-dim (3D data)
#     and applies FFT2 to the (nx, ny) axis.

#     """

#     def __init__(self, noise_lvl=None, norm='ortho'):
#         super(DataConsistencyInKspace, self).__init__()
#         self.normalized = norm == 'ortho'
#         self.noise_lvl = noise_lvl

#     def forward(self, *input, **kwargs):
#         return self.perform(*input)

#     def perform(self, x, k0, mask):
#         """
#         x    - input in image domain, of shape (n, 2, nx, ny[, nt])
#         k0   - initially sampled elements in k-space
#         mask - corresponding nonzero location
#         """

#         if x.dim() == 4: # input is 2D
#             x    = x.permute(0, 2, 3, 1)
#             k0   = k0.permute(0, 2, 3, 1)
#             mask = mask.permute(0, 2, 3, 1)
#         elif x.dim() == 5: # input is 3D
#             x    = x.permute(0, 4, 2, 3, 1)
#             k0   = k0.permute(0, 4, 2, 3, 1)
#             mask = mask.permute(0, 4, 2, 3, 1)
#         # perform-x-shape: torch.Size([4, 30, 256, 32, 2])
#         # perform-x-dtype: torch.float32
#         # print('perform-x-shape:',x.shape)
#         # print('perform-x-dtype:',x.dtype)   
#         # k = torch.fft(x, 2, normalized=self.normalized)
#         # out = data_consistency(k, k0, mask, self.noise_lvl)
#         # x_res = torch.ifft(out, 2, normalized=self.normalized)
        
#         # k = torch.fft.fft2(x, dim=(-2, -1), normalized=self.normalized)
#         k = torch.fft.fft2(x, dim=(-2, -1), norm='forward')
#         # out = data_consistency(k, k0, mask, self.noise_lvl)
#         out = data_consistency(k, k0, mask, self.noise_lvl,'./main_crnn_test','crnn0111')
#         x_res = torch.fft.ifft2(out, dim=(-2, -1), norm='backward')


#         if x.dim() == 4:
#             x_res = x_res.permute(0, 3, 1, 2)
#         elif x.dim() == 5:
#             x_res = x_res.permute(0, 4, 2, 3, 1)
#         # perform-x_res-shape: torch.Size([4, 2, 256, 32, 30])
#         # perform-x_res-dtype: torch.complex64
#         # print('perform-x_res-1-shape:',x_res.shape)
#         # print('perform-x_res-1-dtype:',x_res.dtype)
        
#         # x_res = torch.view_as_real(x_res)  # 将复数拆分为实部和虚部作为两个通道
#         x_res = torch.abs(x_res)  # 获取幅值
#         # perform-x_res-2-shape: torch.Size([1, 2, 256, 256, 30])
#         # perform-x_res-2-dtype: torch.float32
#         # print('perform-x_res-2-shape:',x_res.shape)
#         # print('perform-x_res-2-dtype:',x_res.dtype)     
#         return x_res


def get_add_neighbour_op(nc, frame_dist, divide_by_n, clipped):
    max_sample = max(frame_dist) *2 + 1

    # for non-clipping, increase the input circularly
    if clipped:
        padding = (max_sample//2, 0, 0)
    else:
        padding = 0

    # expect data to be in this format: (n, nc, nt, nx, ny) (due to FFT)
    conv = nn.Conv3d(in_channels=nc, out_channels=nc*len(frame_dist),
                     kernel_size=(max_sample, 1, 1),
                     stride=1, padding=padding, bias=False)

    # Although there is only 1 parameter, need to iterate as parameters return generator
    conv.weight.requires_grad = False

    # kernel has size nc=2, nc'=8, kt, kx, ky
    for i, n in enumerate(frame_dist):
        m = max_sample // 2
        #c = 1 / (n * 2 + 1) if divide_by_n else 1
        c = 1
        wt = np.zeros((2, max_sample, 1, 1), dtype=np.float32)
        wt[0, m-n:m+n+1] = c
        wt2 = np.zeros((2, max_sample, 1, 1), dtype=np.float32)
        wt2[1, m-n:m+n+1] = c

        conv.weight.data[2*i] = torch.from_numpy(wt)
        conv.weight.data[2*i+1] = torch.from_numpy(wt2)

    conv.cuda()
    return conv


class KspaceFillNeighbourLayer(nn.Module):
    '''
    k-space fill layer - The input data is assumed to be in k-space grid.

    The input data is assumed to be in k-space grid.
    This layer should be invoked from AverageInKspaceLayer
    '''
    def __init__(self, frame_dist, divide_by_n=False, clipped=True, **kwargs):
        # frame_dist is the extent that data sharing goes.
        # e.g. current frame is 3, frame_dist = 2, then 1,2, and 4,5 are added for reconstructing 3
        super(KspaceFillNeighbourLayer, self).__init__()
        print("fr_d={}, divide_by_n={}, clippd={}".format(frame_dist, divide_by_n, clipped))
        if 0 not in frame_dist:
            raise ValueError("There suppose to be a 0 in fr_d in config file!")
            frame_dist = [0] + frame_dist # include ID

        self.frame_dist  = frame_dist
        self.n_samples   = [1 + 2*i for i in self.frame_dist]
        self.divide_by_n = divide_by_n
        self.clipped     = clipped
        self.op = get_add_neighbour_op(2, frame_dist, divide_by_n, clipped)

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def perform(self, k, mask):
        '''

        Parameters
        ------------------------------
        inputs: two 5d tensors, [kspace_data, mask], each of shape (n, 2, NT, nx, ny)

        Returns
        ------------------------------
        output: 5d tensor, missing lines of k-space are filled using neighbouring frames.
        shape becomes (n* (len(frame_dist), 2, nt, nx, ny)
        '''
        max_d = max(self.frame_dist)
        k_orig = k
        mask_orig = mask
        if not self.clipped:
            # pad input along nt direction, which is circular boundary condition. Otherwise, just pad outside
            # places with 0 (zero-boundary condition)
            k = torch.cat([k[:,:,-max_d:], k, k[:,:,:max_d]], 2)
            mask = torch.cat([mask[:,:,-max_d:], mask, mask[:,:,:max_d]], 2)

        # start with x, then copy over accumulatedly...
        res = self.op(k)
        if not self.divide_by_n:
            # divide by n basically means for each kspace location, if n non-zero values from neighboring
            # time frames contributes to it, then divide this entry by n (like a normalization)
            res_mask = self.op(mask)
            res = res / res_mask.clamp(min=1)
        else:
            res_mask = self.op(torch.ones_like(mask))
            res = res / res_mask.clamp(min=1)

        res = data_consistency(res,
                               k_orig.repeat(1,len(self.frame_dist),1,1,1),
                               mask_orig.repeat(1,len(self.frame_dist),1,1,1))

        nb, nc_ri, nt, nx, ny = res.shape # here ri_nc is complicated with data sharing replica and real-img dimension
        res = res.reshape(nb, nc_ri//2, 2, nt, nx, ny)
        return res


class AveragingInKspace(nn.Module):
    '''
    Average-in-k-space layer

    First transforms the representation in Fourier domain,
    then performs averaging along temporal axis, then transforms back to image
    domain. Works only for 5D tensor (see parameter descriptions).


    Parameters
    -----------------------------
    incomings: two 5d tensors, [kspace_data, mask], each of shape (n, 2, nx, ny, nt)

    data_shape: shape of the incoming tensors: (n, 2, nx, ny, nt) (This is for convenience)

    frame_dist: a list of distances of neighbours to sample for each averaging channel
        if frame_dist=[1], samples from [-1, 1] for each temporal frames
        if frame_dist=[3, 5], samples from [-3,-2,...,0,1,...,3] for one,
                                           [-5,-4,...,0,1,...,5] for the second one

    divide_by_n: bool - Decides how averaging will be done.
        True => divide by number of neighbours (=#2*frame_dist+1)
        False => divide by number of nonzero contributions

    clipped: bool - By default the layer assumes periodic boundary condition along temporal axis.
        True => Averaging will be clipped at the boundary, no circular references.
        False => Averages with circular referencing (i.e. at t=0, gets contribution from t=nt-1, so on).

    Returns
    ------------------------------
    output: 5d tensor, missing lines of k-space are filled using neighbouring frames.
            shape becomes (n, (len(frame_dist))* 2, nx, ny, nt)
    '''

    def __init__(self, frame_dist, divide_by_n=False, clipped=True, norm='ortho'):
        super(AveragingInKspace, self).__init__()
        self.normalized = norm == 'ortho'
        self.frame_dist = frame_dist
        self.divide_by_n = divide_by_n
        self.kavg = KspaceFillNeighbourLayer(frame_dist, divide_by_n, clipped)

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def perform(self, x, mask):
        """
        x    - input in image space, shape (n, 2, nx, ny, nt)
        mask - corresponding nonzero location
        """
        mask = mask.permute(0, 1, 4, 2, 3)

        x = x.permute(0, 4, 2, 3, 1) # put t to front, in convenience for fft
        k = torch.fft(x, 2, normalized=self.normalized)
        k = k.permute(0, 4, 1, 2, 3) # then put ri to the front, then t

        # data sharing
        # nc is the numpy of copies of kspace, specified by frame_dist
        out = self.kavg.perform(k, mask)
        # after datasharing, it is nb, nc, 2, nt, nx, ny

        nb, nc, _, nt, nx, ny = out.shape # , jo's version

        # out.shape: [nb, 2*len(frame_dist), nt, nx, ny]
        # we then detatch confused real/img channel and replica kspace channel due to datasharing (nc)
        out = out.permute(0,1,3,4,5,2) # jo version, split ri and nc, put ri to the back for ifft
        x_res = torch.ifft(out, 2, normalized=self.normalized)


        # now nb, nc, nt, nx, ny, ri, put ri to channel position, and after nc (i.e. within each nc)
        x_res = x_res.permute(0,1,5,3,4,2).reshape(nb, nc*2, nx,ny, nt)# jo version

        return x_res
    
'''
假设 prefix='last_epoch_'，保存的文件如下：

k-space 数据
last_epoch_k.npy：k-space 数据（.npy 格式）。

last_epoch_k.mat：k-space 数据（.mat 格式）。

last_epoch_k0.npy：初始采样的 k-space 数据（.npy 格式）。

last_epoch_k0.mat：初始采样的 k-space 数据（.mat 格式）。

last_epoch_out.npy：输出的 k-space 数据（.npy 格式）。

last_epoch_out.mat：输出的 k-space 数据（.mat 格式）。

图像域数据
last_epoch_k_image.npy：k-space 转换后的图像域数据（.npy 格式）。

last_epoch_k_image.mat：k-space 转换后的图像域数据（.mat 格式）。

last_epoch_k_image.png：k-space 转换后的图像域数据（图片格式）。

last_epoch_k0_image.npy：k0 转换后的图像域数据（.npy 格式）。

last_epoch_k0_image.mat：k0 转换后的图像域数据（.mat 格式）。

last_epoch_k0_image.png：k0 转换后的图像域数据（图片格式）。

last_epoch_out_image.npy：out 转换后的图像域数据（.npy 格式）。

last_epoch_out_image.mat：out 转换后的图像域数据（.mat 格式）。

last_epoch_out_image.png：out 转换后的图像域数据（图片格式）。

5. 总结
k-space 数据：直接保存为 .mat 和 .npy 格式。
图像域数据：通过逆傅里叶变换转换后，保存为 .mat、.npy 和图片格式。
文件名：通过 prefix 参数区分不同阶段的数据（如 last_epoch_）。
如果你有其他需求（如保存所有 118 张图片的数据），可以进一步调整代码逻辑！
'''