import numpy as np

def r2c(x, axis=1):
    """Convert pseudo-complex data (2 real channels) to complex data

    x: ndarray
        input data
    axis: int
        the axis that is used to represent the real and complex channel.
        e.g. if axis == i, then x.shape looks like (n_1, n_2, ..., n_i-1, 2, n_i+1, ..., nm)
    """
    shape = x.shape
    if axis < 0: axis = x.ndim + axis
    ctype = np.complex64 if x.dtype == np.float32 else np.complex128

    if axis < len(shape):
        newshape = tuple([i for i in range(0, axis)]) \
                   + tuple([i for i in range(axis+1, x.ndim)]) + (axis,)

        x = x.transpose(newshape)

    x = np.ascontiguousarray(x).view(dtype=ctype)
    return x.reshape(x.shape[:-1])


# def c2r(x, axis=1):
#     """Convert complex data to pseudo-complex data (2 real channels)

#     x: ndarray
#         input data
#     axis: int
#         the axis that is used to represent the real and complex channel.
#         e.g. if axis == i, then x.shape looks like (n_1, n_2, ..., n_i-1, 2, n_i+1, ..., nm)
#     """
#     shape = x.shape
#     dtype = np.float32 if x.dtype == np.complex64 else np.float64
#     # dnn_io-c2r-x-shape: torch.Size([2, 256, 256, 30])
#     # dnn_io-c2r-x-dtype: torch.complex64
#     print('dnn_io-c2r-x-shape:',x.shape)
#     print('dnn_io-c2r-x-dtype:',x.dtype)
#     x = np.ascontiguousarray(x).view(dtype=dtype).reshape(shape + (2,))

#     n = x.ndim
#     if axis < 0: axis = n + axis
#     if axis < n:
#         newshape = tuple([i for i in range(0, axis)]) + (n-1,) \
#                    + tuple([i for i in range(axis, n-1)])
#         x = x.transpose(newshape)

#     return x
def c2r(x, axis=1):
    """Convert complex data to pseudo-complex data (2 real channels)

    x: ndarray
        input data
    axis: int
        the axis that is used to represent the real and complex channel.
        e.g. if axis == i, then x.shape looks like (n_1, n_2, ..., n_i-1, 2, n_i+1, ..., nm)
    """
    shape = x.shape
    dtype = np.float32 if x.dtype == np.complex64 else np.float64

    # 确保输入数据是复数类型
    if not np.iscomplexobj(x):
        raise ValueError("Input data must be complex.")

    # 将复数数据转换为两个实数通道
    x_real = np.ascontiguousarray(x.real)
    x_imag = np.ascontiguousarray(x.imag)
    x = np.stack([x_real, x_imag], axis=-1)  # 在最后一个维度上堆叠实部和虚部

    # 调整形状
    n = x.ndim
    if axis < 0: axis = n + axis
    if axis < n:
        newshape = tuple([i for i in range(0, axis)]) + (n-1,) \
                   + tuple([i for i in range(axis, n-1)])
        x = x.transpose(newshape)

    return x

def mask_r2c(m):
    return m[0] if m.ndim == 3 else m[:, 0]


def to_tensor_format(x, mask=False):
    """
    Assumes data is of shape (n[, nt], nx, ny).
    Reshapes to (n, n_channels, nx, ny[, nt])
    Note: Depth must be the last axis, the dimensions will be reordered
    """
    if x.ndim == 4:  # n 3D inputs. reorder axes
        x = np.transpose(x, (0, 2, 3, 1))

    if mask:  # Hacky solution
        x = x*(1+1j)
    # to_tensor_format-x-shape-1: torch.Size([2, 256, 256, 30])
    # to_tensor_format-x-dtype-1: torch.complex64
    # print('to_tensor_format-x-shape-1:',x.shape)
    # print('to_tensor_format-x-dtype-1:',x.dtype)
    x = c2r(x)
    # to_tensor_format-x-shape-2: (1, 2, 256, 256, 30)
    # to_tensor_format-x-dtype-2: float64
    # print('to_tensor_format-x-shape-2:',x.shape)
    # print('to_tensor_format-x-dtype-2:',x.dtype)
    return x


def from_tensor_format(x, mask=False):
    """
    Assumes data is of shape (n, 2, nx, ny[, nt]).
    Reshapes to (n, [nt, ]nx, ny)
    """
    if x.ndim == 5:  # n 3D inputs. reorder axes
        x = np.transpose(x, (0, 1, 4, 2, 3))

    if mask:
        x = mask_r2c(x)
    else:
        x = r2c(x)

    return x
