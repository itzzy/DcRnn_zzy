#!/usr/bin/env python
from __future__ import print_function, division

import os
import time
import torch
import torch.optim as optim
from torch.autograd import Variable
import argparse
import matplotlib.pyplot as plt

from os.path import join
from scipy.io import loadmat

from utils import compressed_sensing as cs
from utils.metric import complex_psnr

from cascadenet_pytorch.model_pytorch import *
from cascadenet_pytorch.dnn_io import to_tensor_format
from cascadenet_pytorch.dnn_io import from_tensor_format


def prep_input(im, acc=4.0):
    """Undersample the batch, then reformat them into what the network accepts.

    Parameters
    ----------
    gauss_ivar: float - controls the undersampling rate.
                        higher the value, more undersampling
    """
    mask = cs.cartesian_mask(im.shape, acc, sample_n=8)
    im_und, k_und = cs.undersample(im, mask, centred=False, norm='ortho')
    im_gnd_l = torch.from_numpy(to_tensor_format(im))
    im_und_l = torch.from_numpy(to_tensor_format(im_und))
    k_und_l = torch.from_numpy(to_tensor_format(k_und))
    mask_l = torch.from_numpy(to_tensor_format(mask, mask=True))

    return im_und_l, k_und_l, mask_l, im_gnd_l


def iterate_minibatch(data, batch_size, shuffle=True):
    n = len(data)

    if shuffle:
        data = np.random.permutation(data)

    for i in range(0, n, batch_size):
        yield data[i:i+batch_size]


def create_dummy_data():
    """Create small cardiac data based on patches for demo.

    Note that in practice, at test time the method will need to be applied to
    the whole volume. In addition, one would need more data to prevent
    overfitting.
    该函数的作用是生成一个基于心脏 MRI 数据的小型数据集，用于演示（demo）目的。
    数据是通过从原始数据中提取小块（patches）生成的，而不是使用完整的 MRI 体积数据。
    在实际应用中，测试时需要将方法应用于整个 MRI 体积数据，而不是仅仅处理小块数据。
    这是因为在实际场景中，MRI 数据通常是完整的 3D 体积（如整个心脏），而不是小块数据。
    此外，为了防止过拟合，实际应用中需要更多的数据。
    当前函数生成的数据量较小，仅用于演示目的。在实际训练中，数据量不足可能导致模型过拟合，因此需要更大规模的数据集。
    """
    data = loadmat(join(project_root, './data/cardiac.mat'))['seq']
    nx, ny, nt = data.shape
    ny_red = 8
    sl = ny//ny_red
    data_t = np.transpose(data, (2, 0, 1))
    
    # Synthesize data by extracting patches
    train = np.array([data_t[..., i:i+sl] for i in np.random.randint(0, sl*3, 20)])
    validate = np.array([data_t[..., i:i+sl] for i in (sl*4, sl*5)])
    test = np.array([data_t[..., i:i+sl] for i in (sl*6, sl*7)])

    return train, validate, test


if __name__ == '__main__':
    # 定义命令行参数，如训练轮数、学习率等。
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', metavar='int', nargs=1, default=['10'],
                        help='number of epochs')
    parser.add_argument('--batch_size', metavar='int', nargs=1, default=['1'],
                        help='batch size')
    parser.add_argument('--lr', metavar='float', nargs=1,
                        default=['0.001'], help='initial learning rate')
    parser.add_argument('--acceleration_factor', metavar='float', nargs=1,
                        default=['4.0'],
                        help='Acceleration factor for k-space sampling')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--savefig', action='store_true',
                        help='Save output images and masks')

    args = parser.parse_args()
    # # 检查是否支持 GPU
    cuda = True if torch.cuda.is_available() else False
    # # 设置张量类型
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    # Project config 
    model_name = 'crnn_mri'
    acc = float(args.acceleration_factor[0])  # undersampling rate
    num_epoch = int(args.num_epoch[0])
    batch_size = int(args.batch_size[0])
    Nx, Ny, Nt = 256, 256, 30
    Ny_red = 8
    save_fig = args.savefig
    save_every = 5

    # Configure directory info 项目路径和数据
    project_root = '.'
    save_dir = join(project_root, 'models/%s' % model_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Create dataset # 加载数据
    train, validate, test = create_dummy_data()

    # Test creating mask and compute the acceleration rate
    dummy_mask = cs.cartesian_mask((10, Nx, Ny//Ny_red), acc, sample_n=8)
    sample_und_factor = cs.undersampling_rate(dummy_mask)
    print('Undersampling Rate: {:.2f}'.format(sample_und_factor))

    # Specify network # 构造 CRNN 网络
    rec_net = CRNN_MRI()
    # 损失函数：均方误差
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(rec_net.parameters(), lr=float(args.lr[0]), betas=(0.5, 0.999))

    # # build CRNN-MRI with pre-trained parameters
    # rec_net.load_state_dict(torch.load('./models/pretrained/crnn_mri_d5_c5.pth'))

    if cuda:
        rec_net = rec_net.cuda()
        criterion.cuda()

    i = 0
    # 训练、验证、测试流程
    for epoch in range(num_epoch):
        t_start = time.time()
        # Training
        train_err = 0
        train_batches = 0
        # # 训练循环
        for im in iterate_minibatch(train, batch_size, shuffle=True):
            # # 准备数据
            im_und, k_und, mask, im_gnd = prep_input(im, acc)
            #  # 转为 Variable
            im_u = Variable(im_und.type(Tensor))
            k_u = Variable(k_und.type(Tensor))
            mask = Variable(mask.type(Tensor))
            gnd = Variable(im_gnd.type(Tensor))

            optimizer.zero_grad()
            rec = rec_net(im_u, k_u, mask, test=False)
            loss = criterion(rec, gnd)
            loss.backward()
            optimizer.step()

            train_err += loss.item()
            train_batches += 1

            if args.debug and train_batches == 20:
                break

        validate_err = 0
        validate_batches = 0
        rec_net.eval()
        for im in iterate_minibatch(validate, batch_size, shuffle=False):
            im_und, k_und, mask, im_gnd = prep_input(im, acc)
            with torch.no_grad():
                im_u = Variable(im_und.type(Tensor))
                k_u = Variable(k_und.type(Tensor))
                mask = Variable(mask.type(Tensor))
                gnd = Variable(im_gnd.type(Tensor))

            pred = rec_net(im_u, k_u, mask, test=True)
            err = criterion(pred, gnd)

            validate_err += err
            validate_batches += 1

            if args.debug and validate_batches == 20:
                break

        vis = []
        test_err = 0
        base_psnr = 0
        test_psnr = 0
        test_batches = 0
        for im in iterate_minibatch(test, batch_size, shuffle=False):
            im_und, k_und, mask, im_gnd = prep_input(im, acc)
            with torch.no_grad():
                im_u = Variable(im_und.type(Tensor))
                k_u = Variable(k_und.type(Tensor))
                mask = Variable(mask.type(Tensor))
                gnd = Variable(im_gnd.type(Tensor))

            pred = rec_net(im_u, k_u, mask, test=True)
            err = criterion(pred, gnd)
            test_err += err
            for im_i, und_i, pred_i in zip(im,
                                           from_tensor_format(im_und.numpy()),
                                           from_tensor_format(pred.data.cpu().numpy())):
                base_psnr += complex_psnr(im_i, und_i, peak='max')
                test_psnr += complex_psnr(im_i, pred_i, peak='max')

            if save_fig and test_batches % save_every == 0:
                vis.append((from_tensor_format(im_gnd.numpy())[0],
                            from_tensor_format(pred.data.cpu().numpy())[0],
                            from_tensor_format(im_und.numpy())[0],
                            from_tensor_format(mask.data.cpu().numpy(), mask=True)[0]))

            test_batches += 1
            if args.debug and test_batches == 20:
                break

        t_end = time.time()

        train_err /= train_batches
        validate_err /= validate_batches
        test_err /= test_batches
        base_psnr /= (test_batches*batch_size)
        test_psnr /= (test_batches*batch_size)

        # Then we print the results for this epoch:
        print("Epoch {}/{}".format(epoch+1, num_epoch))
        print(" time: {}s".format(t_end - t_start))
        print(" training loss:\t\t{:.6f}".format(train_err))
        print(" validation loss:\t{:.6f}".format(validate_err))
        print(" test loss:\t\t{:.6f}".format(test_err))
        print(" base PSNR:\t\t{:.6f}".format(base_psnr))
        print(" test PSNR:\t\t{:.6f}".format(test_psnr))

        # save the model
        if epoch in [1, 2, num_epoch-1]:
            # 结果打印与模型保存
            if save_fig:

                for im_i, pred_i, und_i, mask_i in vis:
                    im = abs(np.concatenate([und_i[0], pred_i[0], im_i[0], im_i[0] - pred_i[0]], 1))
                    plt.imsave(join(save_dir, 'im{0}_x.png'.format(i)), im, cmap='gray')

                    im = abs(np.concatenate([und_i[..., 0], pred_i[..., 0],
                                             im_i[..., 0], im_i[..., 0] - pred_i[..., 0]], 0))
                    plt.imsave(join(save_dir, 'im{0}_t.png'.format(i)), im, cmap='gray')
                    plt.imsave(join(save_dir, 'mask{0}.png'.format(i)),
                               np.fft.fftshift(mask_i[..., 0]), cmap='gray')
                    i += 1

            name = '%s_epoch_%d.npz' % (model_name, epoch)
            #  # 保存模型参数
            torch.save(rec_net.state_dict(), join(save_dir, name))
            print('model parameters saved at %s' % join(os.getcwd(), name))
            print('')
            
'''
运行本脚本300个epoch的部分日志：
CRNNcell-hidden_iteration: torch.float32
CRNNcell-input: torch.float32
CRNNcell-hidden: torch.float32
CRNNcell-hidden_iteration: torch.float32
perform-x-shape: torch.Size([2, 30, 256, 32, 2])
perform-x-dtype: torch.float32
perform-x_res-1-shape: torch.Size([2, 2, 256, 32, 30])
perform-x_res-1-dtype: torch.complex64
perform-x_res-2-shape: torch.Size([2, 2, 256, 32, 30])
perform-x_res-2-dtype: torch.float32
Epoch 300/300
 time: 10.495876789093018s
 training loss:		0.010136
 validation loss:	0.005547
 test loss:		0.011719
 base PSNR:		10.251757
 test PSNR:		6.187429
im_i---------
model parameters saved at /nfs/zzy/code/Deep-MRI-Reconstruction/crnn_mri_epoch_299.npz
'''

'''
这段代码的作用是**将重建结果、原始图像、下采样图像和掩码保存为图像文件**，并生成两种不同的图像：`im{0}_x.png` 和 `im{0}_t.png`。以下是代码的详细解释：

---

### 代码的作用

1. **`vis` 列表**：
   - `vis` 是一个列表，其中每个元素是一个元组 `(im_i, pred_i, und_i, mask_i)`，分别表示：
     - `im_i`：原始图像（ground truth）。
     - `pred_i`：模型重建的图像。
     - `und_i`：下采样后的图像（未重建的欠采样图像）。
     - `mask_i`：下采样时使用的掩码。

2. **保存图像**：
   - 代码通过循环遍历 `vis` 列表中的每个元素，生成并保存以下图像：
     - `im{0}_x.png`：横向拼接的图像，显示空间维度的对比。
     - `im{0}_t.png`：纵向拼接的图像，显示时间维度的对比。
     - `mask{0}.png`：下采样掩码。

3. **`i` 的作用**：
   - `i` 是一个计数器，用于为保存的图像文件生成唯一的文件名（例如 `im0_x.png`, `im1_x.png` 等）。

---

### 代码逐行解析

#### 1. 生成 `im{0}_x.png`
```python
im = abs(np.concatenate([und_i[0], pred_i[0], im_i[0], im_i[0] - pred_i[0]], 1))
plt.imsave(join(save_dir, 'im{0}_x.png'.format(i)), im, cmap='gray')
```
- **`und_i[0]`**：下采样图像的第 0 帧。
- **`pred_i[0]`**：重建图像的第 0 帧。
- **`im_i[0]`**：原始图像的第 0 帧。
- **`im_i[0] - pred_i[0]`**：原始图像与重建图像的残差（误差）。
- **`np.concatenate(..., 1)`**：将以上 4 个图像在水平方向（第 1 维度）拼接。
- **`abs(...)`**：取绝对值（确保图像值为非负数）。
- **`plt.imsave`**：将拼接后的图像保存为 `im{0}_x.png`。

#### 2. 生成 `im{0}_t.png`
```python
im = abs(np.concatenate([und_i[..., 0], pred_i[..., 0],
                         im_i[..., 0], im_i[..., 0] - pred_i[..., 0]], 0))
plt.imsave(join(save_dir, 'im{0}_t.png'.format(i)), im, cmap='gray')
```
- **`und_i[..., 0]`**：下采样图像的第 0 个时间帧（所有空间维度）。
- **`pred_i[..., 0]`**：重建图像的第 0 个时间帧。
- **`im_i[..., 0]`**：原始图像的第 0 个时间帧。
- **`im_i[..., 0] - pred_i[..., 0]`**：原始图像与重建图像的残差。
- **`np.concatenate(..., 0)`**：将以上 4 个图像在垂直方向（第 0 维度）拼接。
- **`abs(...)`**：取绝对值。
- **`plt.imsave`**：将拼接后的图像保存为 `im{0}_t.png`。

#### 3. 生成 `mask{0}.png`
```python
plt.imsave(join(save_dir, 'mask{0}.png'.format(i)),
           np.fft.fftshift(mask_i[..., 0]), cmap='gray')
```
- **`mask_i[..., 0]`**：掩码的第 0 个时间帧。
- **`np.fft.fftshift`**：将掩码的中心移动到图像中心（便于可视化）。
- **`plt.imsave`**：将掩码保存为 `mask{0}.png`。

---

### `im{0}_x.png` 和 `im{0}_t.png` 的区别

| 文件名         | 拼接方向 | 内容维度       | 作用                                   |
|----------------|----------|----------------|----------------------------------------|
| `im{0}_x.png`  | 水平方向 | 空间维度       | 显示某一时间帧的空间信息对比。         |
| `im{0}_t.png`  | 垂直方向 | 时间维度       | 显示某一空间位置的时间信息对比。       |

#### 1. **`im{0}_x.png`**
- **拼接方向**：水平方向（`axis=1`）。
- **内容**：
  - 左起第 1 部分：下采样图像（`und_i[0]`）。
  - 左起第 2 部分：重建图像（`pred_i[0]`）。
  - 左起第 3 部分：原始图像（`im_i[0]`）。
  - 左起第 4 部分：残差图像（`im_i[0] - pred_i[0]`）。
- **作用**：用于对比某一时间帧（例如第 0 帧）的空间信息。

#### 2. **`im{0}_t.png`**
- **拼接方向**：垂直方向（`axis=0`）。
- **内容**：
  - 上起第 1 部分：下采样图像的时间帧（`und_i[..., 0]`）。
  - 上起第 2 部分：重建图像的时间帧（`pred_i[..., 0]`）。
  - 上起第 3 部分：原始图像的时间帧（`im_i[..., 0]`）。
  - 上起第 4 部分：残差图像的时间帧（`im_i[..., 0] - pred_i[..., 0]`）。
- **作用**：用于对比某一空间位置（例如第 0 个像素点）的时间信息。

---

### 总结
- **`im{0}_x.png`**：显示某一时间帧的空间信息对比。
- **`im{0}_t.png`**：显示某一空间位置的时间信息对比。
- **`mask{0}.png`**：显示下采样掩码。

通过这两种图像，可以直观地对比模型的重建效果，分析空间和时间维度上的差异。
'''