__author__ = 'Jo Schlemper'

import numpy as np


# np.abs(x - y)**2: 计算数组 x 和 y 之间差的绝对值的平方。np.mean(...): 计算这些平方差的平均值。
def mse(x, y):
    return np.mean(np.abs(x - y)**2)

# 这个函数计算两个图像 x 和 y 之间的峰值信噪比（PSNR）
def psnr(x, y):
    '''
    Measures the PSNR of recon w.r.t x.
    Image must be of either integer (0, 256) or float value (0,1)
    :param x: [m,n]
    :param y: [m,n]
    :return:
    '''
    assert x.shape == y.shape
    assert x.dtype == y.dtype or np.issubdtype(x.dtype, np.float) \
        and np.issubdtype(y.dtype, np.float)
    if x.dtype == np.uint8:
        max_intensity = 256
    else:
        max_intensity = 1

    mse = np.sum((x - y) ** 2).astype(float) / x.size
    # 使用MSE计算并返回PSNR。
    return 20 * np.log10(max_intensity) - 10 * np.log10(mse)


def complex_psnr(x, y, peak='normalized'):
    '''
    x: reference image
    y: reconstructed image
    peak: normalised or max

    Notice that ``abs'' squares
    Be careful with the order, since peak intensity is taken from the reference
    image (taking from reconstruction yields a different value).

    '''
    mse = np.mean(np.abs(x - y)**2)
    # 如果指定了 peak='max'，则使用参考图像 x 的最大绝对值作为峰值。
    if peak == 'max':
        return 10*np.log10(np.max(np.abs(x))**2/mse)
    else:
        # 如果指定了 peak='normalized'，则使用归一化的峰值（即1）。
        return 10*np.log10(1./mse)
