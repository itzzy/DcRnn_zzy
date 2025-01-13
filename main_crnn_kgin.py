#!/usr/bin/env python2
# -*- coding: utf-8 -*-

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
    # prep_input-mask-shape: (4, 30, 256, 32)
    # prep_input-mask-dtype: float64
    print('prep_input-mask-shape:',mask.shape)
    print('prep_input-mask-dtype:',mask.dtype)
    # 欠采样后的时域数据 和频域数据(kspace)
    im_und, k_und = cs.undersample(im, mask, centred=False, norm='ortho')
    im_gnd_l = torch.from_numpy(to_tensor_format(im))
    im_und_l = torch.from_numpy(to_tensor_format(im_und))
    k_und_l = torch.from_numpy(to_tensor_format(k_und))
    mask_l = torch.from_numpy(to_tensor_format(mask, mask=True))
    # prep_input-mask_l-shape: torch.Size([4, 2, 256, 32, 30])
    # prep_input-mask_l-dtype: torch.float64
    print('prep_input-mask_l-shape:',mask_l.shape)
    print('prep_input-mask_l-dtype:',mask_l.dtype)

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

    """
    '''
    首先使用 loadmat 函数加载位于 project_root 目录下 ./data/cardiac.mat 文件中的数据，并且从中取出名为 'seq' 的数据部分
    （这个 'seq' 应该是在 .mat 文件中定义的变量名，对应着实际的图像数据或者相关序列数据等），将其赋值给 data 变量，
    这里的数据格式应该是 numpy 数组格式，并且其维度信息后续会被使用到。
    '''
    data = loadmat(join(project_root, './data/cardiac.mat'))['seq']
    '''
    解析加载后的数据 data 的形状，获取其在不同维度上的大小，分别赋值给 nx（可能表示图像的某个空间维度大小，比如长度）、
    ny（可能表示宽度等空间维度大小）、nt（可能表示时间维度或者序列维度的大小，例如动态 MRI 数据中的时间帧数等）。
    '''
    nx, ny, nt = data.shape
    # 定义了一个变量 ny_red 并赋值为 8，然后通过 ny 除以 ny_red 计算得到 sl，
    # 这个 sl 可能表示切片的大小或者提取数据块的长度等信息，从后续代码看是用于确定从原始数据中提取数据块的范围。
    ny_red = 8
    sl = ny//ny_red
    # 对原始数据 data 进行维度转置操作，将原来的维度顺序按照 (2, 0, 1) 的顺序重新排列，改变数据的维度顺序，
    # 这可能是为了后续更方便地提取数据块或者符合模型对数据维度的期望顺序等需求。
    data_t = np.transpose(data, (2, 0, 1))
    
    # Synthesize data by extracting patches 合成训练集、验证集和测试集
    '''
    在 data_t 数据上进行操作，使用 np.random.randint(0, sl*3, 20) 随机生成 20 个在 0 到 sl*3 范围内的整数索引 i，
    然后对于每个索引 i，从 data_t 中提取对应的数据块（通过切片操作 data_t[..., i:i+sl]，这里 ... 表示前面维度保持不变，
    只在最后相关维度上进行切片提取数据块），最后将这些提取的数据块组成一个 numpy 数组，作为训练集数据。
    '''
    train = np.array([data_t[..., i:i+sl] for i in np.random.randint(0, sl*3, 20)])
    # 不过这里是直接选取特定的两个索引 sl*4 和 sl*5 对应的两个数据块组成验证集数据，相对来说数据量较少，
    # 用于在训练过程中验证模型的性能等情况。
    validate = np.array([data_t[..., i:i+sl] for i in (sl*4, sl*5)])
    # 选取另外两个特定索引对应的两个数据块组成测试集数据，用于在模型训练完成后评估模型在未见过的数据上的性能表现。
    test = np.array([data_t[..., i:i+sl] for i in (sl*6, sl*7)])

    return train, validate, test

# python main_crnn.py --acceleration_factor 4
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--num_epoch', metavar='int', nargs=1, default=['1'],
    #                     help='number of epochs')
    parser.add_argument('--num_epoch', metavar='int', nargs=1, default=['300'],
                        help='number of epochs')
    # parser.add_argument('--batch_size', metavar='int', nargs=1, default=['1'],
    #                     help='batch size')
    parser.add_argument('--batch_size', metavar='int', nargs=1, default=['4'],
                        help='batch size')
    parser.add_argument('--lr', metavar='float', nargs=1,
                        default=['0.001'], help='initial learning rate')
    parser.add_argument('--acceleration_factor', metavar='float', nargs=1,
                        default=['4.0'],
                        help='Acceleration factor for k-space sampling')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--savefig', action='store_true',default='True',
                        help='Save output images and masks')

    args = parser.parse_args()
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    # 指定使用的GPU编号，这里假设要使用编号为1的GPU（注意GPU编号从0开始计数）
    if cuda:
        torch.cuda.set_device(0)

    # Project config
    model_name = 'crnn_mri'
    acc = float(args.acceleration_factor[0])  # undersampling rate
    num_epoch = int(args.num_epoch[0])
    batch_size = int(args.batch_size[0])
    Nx, Ny, Nt = 256, 256, 30
    Ny_red = 8
    save_fig = args.savefig
    save_every = 5

    # Configure directory info
    project_root = '.'
    save_dir = join(project_root, 'models/%s' % model_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Create dataset
    train, validate, test = create_dummy_data()

    # Test creating mask and compute the acceleration rate
    dummy_mask = cs.cartesian_mask((10, Nx, Ny//Ny_red), acc, sample_n=8)
    sample_und_factor = cs.undersampling_rate(dummy_mask)
    print('Undersampling Rate: {:.2f}'.format(sample_und_factor))

    # Specify network
    rec_net = CRNN_MRI()
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(rec_net.parameters(), lr=float(args.lr[0]), betas=(0.5, 0.999))

    # # build CRNN-MRI with pre-trained parameters
    # rec_net.load_state_dict(torch.load('./models/pretrained/crnn_mri_d5_c5.pth'))

    if cuda:
        rec_net = rec_net.cuda()
        criterion.cuda()

    i = 0
    for epoch in range(num_epoch):
        t_start = time.time()
        # Training
        train_err = 0
        train_batches = 0
        for im in iterate_minibatch(train, batch_size, shuffle=True):
            '''
            im_und: 欠采样的图像数据，大小 [4, 2, 256, 32, 30]。
            k_und: 欠采样的 k 空间 数据，大小与 im_und 相同。
            mask: 欠采样的 k 空间掩码，用于指定哪些位置被采样。
            im_gnd: 原始（完整）的图像数据，作为重建的参考标准。
            '''
            # im-shape: (4, 30, 256, 32)
            # im-dtype: complex64
            print('im-shape:',im.shape)
            print('im-dtype:',im.dtype)
            im_und, k_und, mask, im_gnd = prep_input(im, acc)

            '''
            各个维度的含义
            Batch size (第1维): 4
            表示当前输入是一个批量数据，其中包含 4 个样本。
            这是由 batch_size=4 设置决定的。
            Channels (第2维): 2
            表示数据具有两个通道，可能是 MRI 数据的复数表示。
            通道 1 表示实部，通道 2 表示虚部。
            Height (第3维): 256
            表示 MRI 图像在空间维度上的高度，通常是图像的垂直像素数。
            Width (第4维): 32
            表示图像在宽度方向上的压缩采样（子采样后的宽度）。
            Temporal frames (第5维): 30
            表示时间维度的帧数，即动态 MRI 的时间帧（或序列长度）。
            im_und-shape: torch.Size([4, 2, 256, 32, 30])
            k_und-shape: torch.Size([4, 2, 256, 32, 30])
            mask-shape: torch.Size([4, 2, 256, 32, 30])
            im_gnd-shape: torch.Size([4, 2, 256, 32, 30])
            '''
            # print('im_und-shape:',im_und.shape)
            # print('k_und-shape:',k_und.shape)
            # print('mask-shape:',mask.shape)
            # print('im_gnd-shape:',im_gnd.shape)
            '''
            im_und-dtype: torch.float64
            k_und:-dtype: torch.float64
            mask-dtype: torch.float32
            im_gnd-dtype: torch.float32
            '''
            # print('im_und-dtype:',im_und.dtype)
            # print('k_und:-dtype:',k_und.dtype)
            # print('mask-dtype:',mask.dtype)
            # print('im_gnd-dtype:',im_gnd.dtype)
            im_u = Variable(im_und.type(Tensor))
            k_u = Variable(k_und.type(Tensor))
            mask = Variable(mask.type(Tensor))
            gnd = Variable(im_gnd.type(Tensor))
            # im_u-dtype: torch.float32
            # k_u:-dtype: torch.float32
            # mask-dtype: torch.float32
            # gnd-dtype: torch.float32
            print('im_u-dtype:',im_u.dtype)
            print('k_u:-dtype:',k_u.dtype)
            print('mask-dtype:',mask.dtype)
            print('gnd-dtype:',gnd.dtype)

            optimizer.zero_grad()
            rec = rec_net(im_u, k_u, mask, test=False)
            # main_crnn-rec torch.Size([1, 2, 256, 32, 30])
            # main_crnn-rec-dtype: torch.float32
            print('main_crnn-rec', rec.shape)
            print('main_crnn-rec-dtype:', rec.dtype)
            # print('main_crnn-rec:',rec.dtype)  # 检查输入数据类型
            # print('main_crnn-rec:',rec.shape)  # 检查输入数据类型
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
            # print('save_fig:',save_fig)
            # print('test_batches:',test_batches)
            # print('save_every:',save_every)
            if save_fig and test_batches % save_every == 0:
                # print('vis-append------')
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
        '''
        运行1个epoch打印如下：
        Epoch 1/1
        time: 13.080119132995605s
        training loss:         1156.800310
        validation loss:       0.010139
        test loss:             0.016800
        base PSNR:             20.242637
        test PSNR:             10.799398
        '''
        print("Epoch {}/{}".format(epoch+1, num_epoch))
        print(" time: {}s".format(t_end - t_start))
        print(" training loss:\t\t{:.6f}".format(train_err))
        print(" validation loss:\t{:.6f}".format(validate_err))
        print(" test loss:\t\t{:.6f}".format(test_err))
        print(" base PSNR:\t\t{:.6f}".format(base_psnr))
        print(" test PSNR:\t\t{:.6f}".format(test_psnr))

        # save the model
        if epoch in [1, 2, num_epoch-1]:
            if save_fig:

                for im_i, pred_i, und_i, mask_i in vis:
                    print('im_i---------')
                    im = abs(np.concatenate([und_i[0], pred_i[0], im_i[0], im_i[0] - pred_i[0]], 1))
                    plt.imsave(join(save_dir, 'im{0}_x.png'.format(i)), im, cmap='gray')

                    im = abs(np.concatenate([und_i[..., 0], pred_i[..., 0],
                                             im_i[..., 0], im_i[..., 0] - pred_i[..., 0]], 0))
                    plt.imsave(join(save_dir, 'im{0}_t.png'.format(i)), im, cmap='gray')
                    plt.imsave(join(save_dir, 'mask{0}.png'.format(i)),
                               np.fft.fftshift(mask_i[..., 0]), cmap='gray')
                    i += 1

            name = '%s_epoch_%d.npz' % (model_name, epoch)
            torch.save(rec_net.state_dict(), join(save_dir, name))
            print('model parameters saved at %s' % join(os.getcwd(), name))
            print('')
