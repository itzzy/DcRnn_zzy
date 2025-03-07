import numpy as np
from . import mymath
from numpy.lib.stride_tricks import as_strided
from utils.fastmriBaseUtils import IFFT2c,FFT2c
from numpy.fft import fft2, ifft2, ifftshift, fftshift

def soft_thresh(u, lmda):
    """Soft-threshing operator for complex valued input"""
    Su = (abs(u) - lmda) / abs(u) * u
    Su[abs(u) < lmda] = 0
    return Su


def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)


def var_dens_mask(shape, ivar, sample_high_freq=True):
    """Variable Density Mask (2D undersampling)"""
    if len(shape) == 3:
        Nt, Nx, Ny = shape
    else:
        Nx, Ny = shape
        Nt = 1

    pdf_x = normal_pdf(Nx, ivar)
    pdf_y = normal_pdf(Ny, ivar)
    pdf = np.outer(pdf_x, pdf_y)

    size = pdf.itemsize
    strided_pdf = as_strided(pdf, (Nt, Nx, Ny), (0, Ny * size, size))
    # this must be false if undersampling rate is very low (around 90%~ish)
    if sample_high_freq:
        strided_pdf = strided_pdf / 1.25 + 0.02
    mask = np.random.binomial(1, strided_pdf)

    xc = Nx / 2
    yc = Ny / 2
    mask[:, xc - 4:xc + 5, yc - 4:yc + 5] = True

    if Nt == 1:
        return mask.reshape((Nx, Ny))

    return mask


def cartesian_mask(shape, acc, sample_n=10, centred=False):
    """
    Sampling density estimated from implementation of kt FOCUSS
    shape: tuple - of form (..., nx, ny)
    acc: float - doesn't have to be integer 4, 8, etc..
这段代码实现了一个生成笛卡尔采样掩码（Cartesian Mask）的函数 cartesian_mask，
通常用于磁共振成像（MRI）中的欠采样（undersampling）任务。
掩码的作用是决定在 k 空间（频率域）中哪些数据点被采样，哪些被忽略。以下是对代码的详细解读：
函数功能
目标：生成一个笛卡尔采样掩码，用于模拟 MRI 中的欠采样过程。
输入参数：
shape：掩码的形状，格式为 (..., nx, ny)，其中 nx 和 ny 是 k 空间的尺寸。
acc：加速因子（acceleration factor），控制欠采样的程度。
sample_n：中心区域的采样点数，通常用于保留 k 空间中心的低频信息。
centred：是否将掩码中心化（默认不中心化）。
输出：返回一个与 shape 形状相同的掩码，值为 0 或 1，表示是否采样。
功能：
生成一个笛卡尔采样掩码，用于 MRI 中的欠采样任务。
支持控制加速因子、中心区域采样点数和是否中心化。
核心逻辑：
使用正态分布和均匀分布的混合 PDF 随机选择采样点。
强制保留 k 空间中心区域的采样点。
应用场景：
用于模拟 MRI 中的欠采样过程，生成训练数据或测试数据。
    """
    # N：除了最后两个维度（nx 和 ny）之外的所有维度的乘积。 Nx 和 Ny：k 空间的尺寸（nx 和 ny）。
    N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
    # 生成概率密度函数（PDF）
#     normal_pdf：
# 生成一个正态分布的概率密度函数（PDF），用于控制采样点的分布。
# 正态分布的中心在 Nx/2，标准差为 Nx/10。
    pdf_x = normal_pdf(Nx, 0.5/(Nx/10.)**2) 
    lmda = Nx/(2.*acc) #根据加速因子 acc 计算的一个权重，用于调整采样密度。
    n_lines = int(Nx / acc)

    # add uniform distribution
    pdf_x += lmda * 1./Nx  #在正态分布的基础上，添加一个均匀分布，确保采样点分布更加均匀。
    #处理中心区域 如果指定了 sample_n，则在 k 空间中心区域保留 sample_n 个采样点。
    # 将中心区域的 PDF 值设为 0，避免重复采样。
    if sample_n:
        pdf_x[Nx//2-sample_n//2:Nx//2+sample_n//2] = 0
        # 重新归一化 PDF，确保概率总和为 1。
        pdf_x /= np.sum(pdf_x)
        # 减少需要随机采样的行数 n_lines。
        n_lines -= sample_n
    # 生成掩码
    # 初始化一个形状为 (N, Nx) 的掩码，初始值为 0。
    mask = np.zeros((N, Nx))
    for i in range(N):
        # 根据 PDF 随机选择 n_lines 个采样点，并将掩码中对应位置设为 1。
        idx = np.random.choice(Nx, n_lines, False, pdf_x)
        mask[i, idx] = 1
    # 处理中心区域掩码
    if sample_n:
        mask[:, Nx//2-sample_n//2:Nx//2+sample_n//2] = 1 #如果指定了 sample_n，则在掩码的中心区域强制设置为 1，确保中心区域被采样。
    # 扩展掩码到完整形状
    size = mask.itemsize
    # 将掩码从 (N, Nx) 扩展到 (N, Nx, Ny)，通过在 Ny 维度上复制数据。
    mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))
    # 将掩码调整为输入 shape 的形状。
    mask = mask.reshape(shape)
    # 如果 centred 为 False，则将掩码中心化，使其符合 k空间的默认布局。
    if not centred:
        print('cartesian_mask-centtred:',centred)
        mask = mymath.ifftshift(mask, axes=(-1, -2))

    return mask


# def shear_grid_mask(shape, acceleration_rate, sample_low_freq=True,
#                     centred=False, sample_n=10):
#     '''
#     Creates undersampling mask which samples in sheer grid

#     Parameters
#     ----------

#     shape: (nt, nx, ny)

#     acceleration_rate: int

#     Returns
#     -------

#     array

#     '''
#     Nt, Nx, Ny = shape
#     start = np.random.randint(0, acceleration_rate)
#     mask = np.zeros((Nt, Nx))
#     for t in xrange(Nt):
#         mask[t, (start+t)%acceleration_rate::acceleration_rate] = 1

#     xc = Nx / 2
#     xl = sample_n / 2
#     if sample_low_freq and centred:
#         xh = xl
#         if sample_n % 2 == 0:
#             xh += 1
#         mask[:, xc - xl:xc + xh+1] = 1

#     elif sample_low_freq:
#         xh = xl
#         if sample_n % 2 == 1:
#             xh -= 1

#         if xl > 0:
#             mask[:, :xl] = 1
#         if xh > 0:
#             mask[:, -xh:] = 1

#     mask_rep = np.repeat(mask[..., np.newaxis], Ny, axis=-1)
#     return mask_rep


# def perturbed_shear_grid_mask(shape, acceleration_rate, sample_low_freq=True,
#                               centred=False,
#                               sample_n=10):
#     Nt, Nx, Ny = shape
#     start = np.random.randint(0, acceleration_rate)
#     mask = np.zeros((Nt, Nx))
#     for t in xrange(Nt):
#         mask[t, (start+t)%acceleration_rate::acceleration_rate] = 1

#     # brute force
#     rand_code = np.random.randint(0, 3, size=Nt*Nx)
#     shift = np.array([-1, 0, 1])[rand_code]
#     new_mask = np.zeros_like(mask)
#     for t in xrange(Nt):
#         for x in xrange(Nx):
#             if mask[t, x]:
#                 new_mask[t, (x + shift[t*x])%Nx] = 1

#     xc = Nx / 2
#     xl = sample_n / 2
#     if sample_low_freq and centred:
#         xh = xl
#         if sample_n % 2 == 0:
#             xh += 1
#         new_mask[:, xc - xl:xc + xh+1] = 1

#     elif sample_low_freq:
#         xh = xl
#         if sample_n % 2 == 1:
#             xh -= 1

#         new_mask[:, :xl] = 1
#         new_mask[:, -xh:] = 1
#     mask_rep = np.repeat(new_mask[..., np.newaxis], Ny, axis=-1)

#     return mask_rep

#来自: https://github.com/cq615/kt-Dynamic-MRI-Reconstruction/blob/master/utils/compressed_sensing.py#L5
# def shear_grid_mask(shape, acceleration_rate, sample_low_freq=False,
#                     centred=False, sample_n=4, test=False):
def shear_grid_mask(shape, acceleration_rate, sample_low_freq=False,
                    centred=False, sample_n=10, test=False):
    '''
    Creates undersampling mask which samples in sheer grid

    Parameters
    ----------

    shape: (nt, nx, ny)

    acceleration_rate: int

    Returns
    -------

    array

    '''
    Nt, Nx, Ny = shape
    print('shear_grid_mask-shape:',shape)
    if test:
        start = 0
    else:
        start = np.random.randint(0, acceleration_rate)
    mask = np.zeros((Nt, Nx))
    for t in range(Nt):
        mask[t, (start+t)%acceleration_rate::acceleration_rate] = 1

    xc = Nx // 2
    xl = sample_n // 2
    if sample_low_freq and centred:
        xh = xl
        if sample_n % 2 == 0:
            xh += 1
        mask[:, xc - xl:xc + xh+1] = 1

    elif sample_low_freq:
        xh = xl
        if sample_n % 2 == 1:
            xh -= 1

        if xl > 0:
            mask[:, :xl] = 1
        if xh > 0:
            mask[:, -xh:] = 1

    mask_rep = np.repeat(mask[..., np.newaxis], Ny, axis=-1)
    return mask_rep

def undersample(x, mask, centred=False, norm='ortho', noise=0):
    '''
    Undersample x. FFT2 will be applied to the last 2 axis
    Parameters
    ----------
    x: array_like
        data
    mask: array_like
        undersampling mask in fourier domain

    norm: 'ortho' or None
        if 'ortho', performs unitary transform, otherwise normal dft

    noise_power: float
        simulates acquisition noise, complex AWG noise.
        must be percentage of the peak signal

    Returns
    -------
    xu: array_like
        undersampled image in image domain. Note that it is complex valued

    x_fu: array_like
        undersampled data in k-space

    '''
    assert x.shape == mask.shape
    # zero mean complex Gaussian noise
    noise_power = noise
    nz = np.sqrt(.5)*(np.random.normal(0, 1, x.shape) + 1j * np.random.normal(0, 1, x.shape))
    nz = nz * np.sqrt(noise_power)

    if norm == 'ortho':
        # multiplicative factor
        nz = nz * np.sqrt(np.prod(mask.shape[-2:]))
    else:
        nz = nz * np.prod(mask.shape[-2:])

    if centred:
        axes = (-2, -1)
        x_f = fftshift(fft2(ifftshift(x, axes=axes), norm=norm), axes=axes)
        x_fu = mask * (x_f + nz)
        x_u = fftshift(ifft2(ifftshift(x_fu, axes=axes), norm=norm), axes=axes)
        return x_u, x_fu
    else:
        x_f = fft2(x, norm=norm)
        x_fu = mask * (x_f + nz)
        x_u = ifft2(x_fu, norm=norm)
        return x_u, x_fu

# def undersample(x, mask, centred=False, norm='ortho', noise=0):
#     '''
#     Undersample x. FFT2 will be applied to the last 2 axis
#     Parameters
#     ----------
#     x: array_like
#         data
#     mask: array_like
#         undersampling mask in fourier domain

#     norm: 'ortho' or None
#         if 'ortho', performs unitary transform, otherwise normal dft
#     noise_power: float
#         simulates acquisition noise, complex AWG noise.
#         must be percentage of the peak signal
#     Returns
#     -------
#     xu: array_like
#         undersampled image in image domain. Note that it is complex valued

#     x_fu: array_like
#         undersampled data in k-space
#     '''
#     # undersample-mask-dtype: float64
#     # undersample-mask-mask: (1, 30, 256, 256)
#     # print('undersample-x-dtype:',x.dtype)
#     # print('undersample-x-shape:',x.shape) #undersample-x-shape: (1, 30, 256, 256)
#     # print('undersample-mask-dtype:',mask.dtype)
#     # print('undersample-mask-mask:',mask.shape)
#     assert x.shape == mask.shape
#     # zero mean complex Gaussian noise
#     noise_power = noise
#     nz = np.sqrt(.5)*(np.random.normal(0, 1, x.shape) + 1j * np.random.normal(0, 1, x.shape))
#     nz = nz * np.sqrt(noise_power)

#     if norm == 'ortho':
#         # multiplicative factor
#         nz = nz * np.sqrt(np.prod(mask.shape[-2:]))
#     else:
#         nz = nz * np.prod(mask.shape[-2:])
#     # undersample-nz-dtype: complex128
#     # print('undersample-nz-dtype:',nz.dtype)
#     # print('undersample-nz:',nz)
#     if centred:
#         x_f = mymath.fft2c(x, norm=norm)
#         x_fu = mask * (x_f + nz)
#         x_u = mymath.ifft2c(x_fu, norm=norm)
#         return x_u, x_fu
#     else:
#         x_f = mymath.fft2(x, norm=norm)
#         x_fu = mask * (x_f + nz)
#         x_u = mymath.ifft2(x_fu, norm=norm)
#     # kspace中心化x_fu
#     # x_fu= np.fft.fftshift(x_fu)
#     return x_u, x_fu

# def undersample(x, mask, centred=False, norm='ortho', noise=0):
#     '''
#     Undersample x. FFT2 will be applied to the last 2 axis
#     Parameters
#     ----------
#     x: array_like
#         data
#     mask: array_like
#         undersampling mask in fourier domain

#     norm: 'ortho' or None
#         if 'ortho', performs unitary transform, otherwise normal dft

#     noise_power: float
#         simulates acquisition noise, complex AWG noise.
#         must be percentage of the peak signal

#     Returns
#     -------
#     xu: array_like
#         undersampled image in image domain. Note that it is complex valued

#     x_fu: array_like
#         undersampled data in k-space

#     '''
#     # print('undersample-x-shape:',x.shape) #undersample-x-shape: (4, 18, 192, 192)
#     assert x.shape == mask.shape
#     # zero mean complex Gaussian noise
#     noise_power = noise
#     nz = np.sqrt(.5)*(np.random.normal(0, 1, x.shape) + 1j * np.random.normal(0, 1, x.shape))
#     nz = nz * np.sqrt(noise_power)

#     if norm == 'ortho':
#         # multiplicative factor
#         nz = nz * np.sqrt(np.prod(mask.shape[-2:]))
#     else:
#         nz = nz * np.prod(mask.shape[-2:])

#     if centred:
#         x_f = mymath.fft2c(x, norm=norm)
#         x_fu = mask * (x_f + nz)
#         x_u = mymath.ifft2c(x_fu, norm=norm)
#         return x_u, x_fu
#     else:
#         x_f = FFT2c(x)
#         x_fu = mask * (x_f + nz)
#         x_u = IFFT2c(x_fu)
#         # 函数最终会返回欠采样后的图像数据（在图像域）和欠采样的数据（在 k 空间）
#         return x_u, x_fu


def data_consistency(x, y, mask, centered=False, norm='ortho'):
    '''
    x is in image space,
    y is in k-space
    '''
    if centered:
        xf = mymath.fft2c(x, norm=norm)
        xm = (1 - mask) * xf + y
        xd = mymath.ifft2c(xm, norm=norm)
    else:
        xf = mymath.fft2(x, norm=norm)
        xm = (1 - mask) * xf + y
        xd = mymath.ifft2(xm, norm=norm)

    return xd


def get_phase(x):
    xr = np.real(x)
    xi = np.imag(x)
    phase = np.arctan(xi / (xr + 1e-12))
    return phase


def undersampling_rate(mask):
    return float(mask.sum()) / mask.size
