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
from scipy.io import savemat
# from torch.cuda.amp import autocast, GradScaler
from torch.cuda.amp import GradScaler
from torch.amp import autocast
import numpy as np


from utils import compressed_sensing as cs
from utils.metric import complex_psnr
from utils.model_related import count_parameters
from utils.common_utils import *


from cascadenet_pytorch.model_pytorch import *
from cascadenet_pytorch.dnn_io import to_tensor_format
from cascadenet_pytorch.dnn_io import from_tensor_format
# 导入 TensorBoard 模块
from torch.utils.tensorboard import SummaryWriter


# PyTorch建议在使用多线程时设置OMP_NUM_THREADS环境变量，以避免系统过载。
os.environ['OMP_NUM_THREADS'] = '1'
# 设置PYTORCH_CUDA_ALLOC_CONF环境变量，以减少CUDA内存碎片
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
# os.environ["CUDA_VISIBLE_DEVICES"] = "3" #,0,1,2,4,5,6,7
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 指定使用 GPU 1 和 GPU 4
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'  # 指定使用 GPU 1 和 GPU 4
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 指定使用 GPU 1 和 GPU 4

def prep_input(im, acc=4.0):
    """Undersample the batch, then reformat them into what the network accepts.

    Parameters
    ----------
    gauss_ivar: float - controls the undersampling rate.
                        higher the value, more undersampling
    """
    mask = cs.cartesian_mask(im.shape, acc, sample_n=8)
    # prep_input-mask-shape: (4, 30, 256,256 )
    # prep_input-mask-dtype: float64
    # print('prep_input-mask-shape:',mask.shape)
    # print('prep_input-mask-dtype:',mask.dtype)
    im_und, k_und = cs.undersample(im, mask, centred=False, norm='ortho')
    # im_gnd_l = torch.from_numpy(to_tensor_format(im))
    # im_und_l = torch.from_numpy(to_tensor_format(im_und))
    # k_und_l = torch.from_numpy(to_tensor_format(k_und))
    # mask_l = torch.from_numpy(to_tensor_format(mask, mask=True))
    # 将数据转换为 torch.float32 类型，减少内存占用
    im_gnd_l = torch.from_numpy(to_tensor_format(im)).float()
    im_und_l = torch.from_numpy(to_tensor_format(im_und)).float()
    k_und_l = torch.from_numpy(to_tensor_format(k_und)).float()
    mask_l = torch.from_numpy(to_tensor_format(mask, mask=True)).float()
    # prep_input-mask_l-shape: torch.Size([4, 2, 256, 32, 30])
    # prep_input-mask_l-dtype: torch.float64
    # print('prep_input-mask_l-shape:',mask_l.shape)
    # print('prep_input-mask_l-dtype:',mask_l.dtype)

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
    data = loadmat(join(project_root, './data/cardiac.mat'))['seq']
    nx, ny, nt = data.shape  # 原始数据形状: [30, 256, 256]
    
    # 转置数据，将时间维度放在最前面
    data_t = np.transpose(data, (2, 0, 1))  # 转置后形状: [256, 256, 30]
    
    # 生成训练集、验证集和测试集
    # 保持后两个维度为 [256, 256]，仅从时间维度切片
    train = np.array([data_t for _ in range(20)])  # 训练集: [20, 256, 256, 30]
    validate = np.array([data_t for _ in range(2)])  # 验证集: [2, 256, 256, 30]
    test = np.array([data_t for _ in range(2)])  # 测试集: [2, 256, 256, 30]

    return train, validate, test



# nohup python main_crnn_test.py --acceleration_factor 4 > output_0111.log 2>&1 &
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', metavar='int', nargs=1, default=['1'],
                        help='number of epochs')
    # parser.add_argument('--num_epoch', metavar='int', nargs=1, default=['400'],
    #                     help='number of epochs')
    # parser.add_argument('--batch_size', metavar='int', nargs=1, default=['1'],
    #                     help='batch size')
    parser.add_argument('--batch_size', metavar='int', nargs=1, default=['1'],
                        help='batch size')
    # parser.add_argument('--lr', metavar='float', nargs=1,
    #                     default=['0.001'], help='initial learning rate')
    parser.add_argument('--lr', metavar='float', nargs=1,
                        default=['0.00005'], help='initial learning rate')
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
    model_name = 'crnn_mri_0112'
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
    
    # 初始化 TensorBoard 的 SummaryWriter
    writer = SummaryWriter(log_dir=join(project_root, 'runs', model_name))

    # Create dataset
    train, validate, test = create_dummy_data()
    data_shape = [Nx, Ny, Nt]
    # Test creating mask and compute the acceleration rate
    dummy_mask = cs.cartesian_mask(data_shape, acc, sample_n=8)
    # dummy_mask = cs.cartesian_mask((10, Nx, Ny//Ny_red), acc, sample_n=8)
    # sample_und_factor = cs.undersampling_rate(dummy_mask)
    # # Undersampling Rate: 0.25
    # print('Undersampling Rate: {:.2f}'.format(sample_und_factor))

    # Specify network
    # rec_net = CRNN_MRI()
    # 初始化模型和优化器
    rec_net = CRNN_MRI().cuda()
    print("Parameter Count: %d" % count_parameters(rec_net))
    # 确保模型参数是 FP32
    rec_net.float()

    criterion = torch.nn.MSELoss()
    # optimizer = optim.Adam(rec_net.parameters(), lr=float(args.lr[0]), betas=(0.5, 0.999))
    #使用混合精度训练
    # scaler = GradScaler()
    # scaler = torch.amp.GradScaler('cuda')
    scaler = torch.amp.GradScaler()

    
    # # build CRNN-MRI with pre-trained parameters
    # rec_net.load_state_dict(torch.load('./models/pretrained/crnn_mri_d5_c5.pth'))

    if cuda:
        rec_net = rec_net.cuda()
        criterion.cuda()

    i = 0
    # 定义梯度累积的步数
    accumulation_steps = 4  # 可以根据显存大小调整
    optimizer = optim.Adam(rec_net.parameters(), lr=float(args.lr[0]) * accumulation_steps, betas=(0.5, 0.999))
    
    for epoch in range(num_epoch):
        t_start = time.time()
        # Training
        train_err = 0
        train_batches = 0
        rec_net.train()

        # 获取总 batch 数
        total_batches_train = len(train) // batch_size
        total_batches_validate = len(validate) // batch_size
        total_batches_test = len(test) // batch_size

        for batch_idx, im in enumerate(iterate_minibatch(train, batch_size, shuffle=True)):
            im_und, k_und, mask, im_gnd = prep_input(im, acc)

            im_u = Variable(im_und.type(Tensor))
            k_u = Variable(k_und.type(Tensor))
            mask = Variable(mask.type(Tensor))
            gnd = Variable(im_gnd.type(Tensor))

            # 检查是否是最后一个 epoch 的最后一个 batch
            # is_last_epoch = (epoch == num_epoch - 1)
            # is_last_batch = (batch_idx == total_batches - 1)
            # save_last = is_last_epoch and is_last_batch
            
            # 前向传播
            optimizer.zero_grad()
            with autocast('cuda'):
                # rec = rec_net(im_u, k_u, mask, test=False, save_last=save_last)
                rec = rec_net(im_u, k_u, mask)
                loss = criterion(rec, gnd)

            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 记录训练损失
            train_err += loss.item()
            train_batches += 1
            writer.add_scalar('Loss/Train', loss.item(), epoch * len(train) + train_batches)

            # 释放内存
            torch.cuda.empty_cache()
            del im_u, k_u, mask, gnd, rec, loss

            if args.debug and train_batches == 50:
                break

        # 记录每个 epoch 的平均训练损失
        writer.add_scalar('Loss/Train_Avg', train_err / train_batches, epoch)

        # 验证和测试逻辑保持不变
        validate_err = 0
        validate_batches = 0
        rec_net.eval()
        # for im in iterate_minibatch(validate, batch_size, shuffle=False):
        for batch_idx, im in enumerate(iterate_minibatch(validate, batch_size, shuffle=False)):
            im_und, k_und, mask, im_gnd = prep_input(im, acc)
            
            # 检查是否是最后一个 epoch 的最后一个 batch
            is_last_epoch = (epoch == num_epoch - 1)
            is_last_batch = (batch_idx == total_batches_validate - 1)
            save_last = is_last_epoch and is_last_batch
            print('iterate_minibatch-validate-save_last:',save_last)
            
            with torch.no_grad():
                im_u = Variable(im_und.type(Tensor))
                k_u = Variable(k_und.type(Tensor))
                mask = Variable(mask.type(Tensor))
                gnd = Variable(im_gnd.type(Tensor))
                pred = rec_net(im_u, k_u, mask, test=True)
                # pred = rec_net(im_u, k_u, mask, test=True, save_last=save_last)

            err = criterion(pred, gnd)
            validate_err += err
            validate_batches += 1
            writer.add_scalar('Loss/Validate', err.item(), epoch * len(validate) + validate_batches)

            if args.debug and validate_batches == 20:
                break

        # 记录每个 epoch 的平均验证损失
        writer.add_scalar('Loss/Validate_Avg', validate_err / validate_batches, epoch)

        # 测试逻辑保持不变
        vis = []
        test_err = 0
        base_psnr = 0
        test_psnr = 0
        test_batches = 0
        # for im in iterate_minibatch(test, batch_size, shuffle=False):
        for batch_idx, im in enumerate(iterate_minibatch(test, batch_size, shuffle=True)):
            im_und, k_und, mask, im_gnd = prep_input(im, acc)
            with torch.no_grad():
                im_u = Variable(im_und.type(Tensor))
                k_u = Variable(k_und.type(Tensor))
                mask = Variable(mask.type(Tensor))
                gnd = Variable(im_gnd.type(Tensor))
                 # 检查是否是最后一个 epoch 的最后一个 batch
                is_last_epoch = (epoch == num_epoch - 1)
                is_last_batch = (batch_idx == total_batches_test - 1)
                save_last = is_last_epoch and is_last_batch
                print('iterate_minibatch-test-save_last:',save_last)
                # pred = rec_net(im_u, k_u, mask, test=True)
                rec = rec_net(im_u, k_u, mask, test=True, save_last=save_last)

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
            writer.add_scalar('Loss/Test', err.item(), epoch * len(test) + test_batches)

            if args.debug and test_batches == 50:
                break

        # 记录每个 epoch 的平均测试损失
        writer.add_scalar('Loss/Test_Avg', test_err / test_batches, epoch)
        writer.add_scalar('PSNR/Base', base_psnr / (test_batches * batch_size), epoch)
        writer.add_scalar('PSNR/Test', test_psnr / (test_batches * batch_size), epoch)

        t_end = time.time()

        train_err /= train_batches
        validate_err /= validate_batches
        test_err /= test_batches
        base_psnr /= (test_batches * batch_size)
        test_psnr /= (test_batches * batch_size)

        print("Epoch {}/{}".format(epoch + 1, num_epoch))
        print(" time: {}s".format(t_end - t_start))
        print(" training loss:\t\t{:.6f}".format(train_err))
        print(" validation loss:\t{:.6f}".format(validate_err))
        print(" test loss:\t\t{:.6f}".format(test_err))
        print(" base PSNR:\t\t{:.6f}".format(base_psnr))
        print(" test PSNR:\t\t{:.6f}".format(test_psnr))

        # 保存模型
        if epoch in [1, 2, num_epoch - 1]:
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
            torch.save(rec_net.state_dict(), join(save_dir, name))
            print('model parameters saved at %s' % join(os.getcwd(), name))

    # 训练结束后关闭 SummaryWriter
    writer.close()
    
    # for epoch in range(num_epoch):
    #     t_start = time.time()
    #     # Training
    #     train_err = 0
    #     train_batches = 0
    #     # 训练
    #     rec_net.train()
    #     for im in iterate_minibatch(train, batch_size, shuffle=True):
    #         '''
    #         im_und: 欠采样的图像数据，大小 [4, 2, 256, 32, 30]。
    #         k_und: 欠采样的 k 空间 数据，大小与 im_und 相同。
    #         mask: 欠采样的 k 空间掩码，用于指定哪些位置被采样。
    #         im_gnd: 原始（完整）的图像数据，作为重建的参考标准。
    #         '''
    #         # im-shape: (4, 30, 256, 32)
    #         # im-dtype: complex64
    #         # print('im-shape:',im.shape)
    #         # print('im-dtype:',im.dtype)
    #         im_und, k_und, mask, im_gnd = prep_input(im, acc)

    #         '''
    #         各个维度的含义
    #         Batch size (第1维): 4
    #         表示当前输入是一个批量数据，其中包含 4 个样本。
    #         这是由 batch_size=4 设置决定的。
    #         Channels (第2维): 2
    #         表示数据具有两个通道，可能是 MRI 数据的复数表示。
    #         通道 1 表示实部，通道 2 表示虚部。
    #         Height (第3维): 256
    #         表示 MRI 图像在空间维度上的高度，通常是图像的垂直像素数。
    #         Width (第4维): 32
    #         表示图像在宽度方向上的压缩采样（子采样后的宽度）。
    #         Temporal frames (第5维): 30
    #         表示时间维度的帧数，即动态 MRI 的时间帧（或序列长度）。
    #         im_und-shape: torch.Size([4, 2, 256, 32, 30])
    #         k_und-shape: torch.Size([4, 2, 256, 32, 30])
    #         mask-shape: torch.Size([4, 2, 256, 32, 30])
    #         im_gnd-shape: torch.Size([4, 2, 256, 32, 30])
    #         '''
    #         # print('im_und-shape:',im_und.shape)
    #         # print('k_und-shape:',k_und.shape)
    #         # print('mask-shape:',mask.shape)
    #         # print('im_gnd-shape:',im_gnd.shape)
    #         '''
    #         im_und-dtype: torch.float64
    #         k_und:-dtype: torch.float64
    #         mask-dtype: torch.float32
    #         im_gnd-dtype: torch.float32
    #         '''
    #         # print('im_und-dtype:',im_und.dtype)
    #         # print('k_und:-dtype:',k_und.dtype)
    #         # print('mask-dtype:',mask.dtype)
    #         # print('im_gnd-dtype:',im_gnd.dtype)
    #         im_u = Variable(im_und.type(Tensor))
    #         k_u = Variable(k_und.type(Tensor))
    #         mask = Variable(mask.type(Tensor))
    #         gnd = Variable(im_gnd.type(Tensor))
    #         # im_u = im_u[..., :5]  # 只取前 10 帧
    #         # k_u = k_u[..., :5]
    #         # mask = mask[..., :5]
    #         # gnd = gnd[..., :5]
            
    #         # 在训练循环中，添加以下代码
    #         if epoch == 0 and train_batches == 0:  # 只在第一个 batch 的第一个 epoch 保存
    #             # 将 Tensor 转换为 numpy 数组
    #             im_u_np = im_u.cpu().numpy()
    #             k_u_np = k_u.cpu().numpy()
    #             mask_np = mask.cpu().numpy()

    #             # 保存为 .mat 格式
    #             savemat(join(save_dir, 'im_u.mat'), {'im_u': im_u_np})
    #             savemat(join(save_dir, 'k_u.mat'), {'k_u': k_u_np})
    #             savemat(join(save_dir, 'mask.mat'), {'mask': mask_np})

    #             # 保存为 .npy 格式
    #             np.save(join(save_dir, 'im_u.npy'), im_u_np)
    #             np.save(join(save_dir, 'k_u.npy'), k_u_np)
    #             np.save(join(save_dir, 'mask.npy'), mask_np)

    #             # 保存为图片
    #             plt.imsave(join(save_dir, 'im_u.png'), np.abs(im_u_np[0, 0, :, :, 0]), cmap='gray')
    #             plt.imsave(join(save_dir, 'k_u.png'), np.abs(k_u_np[0, 0, :, :, 0]), cmap='gray')
    #             plt.imsave(join(save_dir, 'mask.png'), np.abs(mask_np[0, 0, :, :, 0]), cmap='gray')

    #             print(f"Saved im_u, k_u, and mask to {save_dir}")
            
    #         # im_u-dtype: torch.float32
    #         # k_u:-dtype: torch.float32
    #         # mask-dtype: torch.float32
    #         # gnd-dtype: torch.float32
    #         # print('im_u-dtype:',im_u.dtype)
    #         # print('k_u:-dtype:',k_u.dtype)
    #         # print('mask-dtype:',mask.dtype)
    #         # print('gnd-dtype:',gnd.dtype)

    #         # optimizer.zero_grad()
    #         # rec = rec_net(im_u, k_u, mask, test=False)
    #         # # main_crnn-rec torch.Size([1, 2, 256, 32, 30])
    #         # # main_crnn-rec-dtype: torch.float32
    #         # print('main_crnn-rec', rec.shape)
    #         # print('main_crnn-rec-dtype:', rec.dtype)
    #         # # print('main_crnn-rec:',rec.dtype)  # 检查输入数据类型
    #         # # print('main_crnn-rec:',rec.shape)  # 检查输入数据类型
    #         # loss = criterion(rec, gnd)
    #         # loss.backward()
    #         # optimizer.step()
    #         # 前向传播
    #         # with autocast():
    #         optimizer.zero_grad()
    #         with autocast('cuda'):  # 修改此处
    #             rec = rec_net(im_u, k_u, mask, test=False)
    #             loss = criterion(rec, gnd)
    #         # torch.cuda.empty_cache()
    #         # 反向传播（累积梯度）
    #         scaler.scale(loss).backward()
    #         # 每 accumulation_steps 步更新一次模型参数
    #         # if (i + 1) % accumulation_steps == 0:
    #         #     # 更新模型参数
    #         #     scaler.step(optimizer)
    #         #     scaler.update()
    #         #     optimizer.zero_grad()  # 清空梯度

    #         #     # 记录训练损失
    #         #     train_err += loss.item()
    #         #     train_batches += 1
    #         #     writer.add_scalar('Loss/Train', loss.item(), epoch * len(train) + train_batches)
    #         # 更新模型参数
    #         scaler.step(optimizer)
    #         scaler.update()
    #         # optimizer.zero_grad()  # 清空梯度

    #             # 记录训练损失
    #         train_err += loss.item()
    #         train_batches += 1
    #         writer.add_scalar('Loss/Train', loss.item(), epoch * len(train) + train_batches)

    #         # scaler.step(optimizer)
    #         # scaler.update()
            
    #         # 释放内存
    #         torch.cuda.empty_cache()
    #         # 可以考虑添加 del 关键字删除不再使用的张量
    #         del im_u, k_u, mask, gnd, rec, loss

    #         # train_err += loss.item()
    #         # train_batches += 1
    #         # # 记录训练损失
    #         # writer.add_scalar('Loss/Train', loss.item(), epoch * len(train) + train_batches)

    #         # if args.debug and train_batches == 20:
    #         if args.debug and train_batches == 50:
    #             break
    #     # 记录每个 epoch 的平均训练损失
    #     writer.add_scalar('Loss/Train_Avg', train_err / train_batches, epoch)
        
    #     validate_err = 0
    #     validate_batches = 0
    #     rec_net.eval()
    #     for im in iterate_minibatch(validate, batch_size, shuffle=False):
    #         im_und, k_und, mask, im_gnd = prep_input(im, acc)
    #         with torch.no_grad():
    #             im_u = Variable(im_und.type(Tensor))
    #             k_u = Variable(k_und.type(Tensor))
    #             mask = Variable(mask.type(Tensor))
    #             gnd = Variable(im_gnd.type(Tensor))
    #             pred = rec_net(im_u, k_u, mask, test=True)
    #         # torch.cuda.empty_cache()
    #         # 
    #         torch.cuda.empty_cache()
    #         err = criterion(pred, gnd)

    #         validate_err += err
    #         validate_batches += 1
    #         # 记录验证损失
    #         writer.add_scalar('Loss/Validate', err.item(), epoch * len(validate) + validate_batches)


    #         if args.debug and validate_batches == 20:
    #             break
    #     # 记录每个 epoch 的平均验证损失
    #     writer.add_scalar('Loss/Validate_Avg', validate_err / validate_batches, epoch)
        
    #     vis = []
    #     test_err = 0
    #     base_psnr = 0
    #     test_psnr = 0
    #     test_batches = 0
    #     for im in iterate_minibatch(test, batch_size, shuffle=False):
    #         im_und, k_und, mask, im_gnd = prep_input(im, acc)
    #         with torch.no_grad():
    #             im_u = Variable(im_und.type(Tensor))
    #             k_u = Variable(k_und.type(Tensor))
    #             mask = Variable(mask.type(Tensor))
    #             gnd = Variable(im_gnd.type(Tensor))
    #             pred = rec_net(im_u, k_u, mask, test=True)
    #         # torch.cuda.empty_cache()
            
    #         # 
    #         torch.cuda.empty_cache()
    #         err = criterion(pred, gnd)
    #         test_err += err
    #         for im_i, und_i, pred_i in zip(im,
    #                                        from_tensor_format(im_und.numpy()),
    #                                        from_tensor_format(pred.data.cpu().numpy())):
    #             base_psnr += complex_psnr(im_i, und_i, peak='max')
    #             test_psnr += complex_psnr(im_i, pred_i, peak='max')
    #         # print('save_fig:',save_fig)
    #         # print('test_batches:',test_batches)
    #         # print('save_every:',save_every)
    #         if save_fig and test_batches % save_every == 0:
    #             # print('vis-append------')
    #             vis.append((from_tensor_format(im_gnd.numpy())[0],
    #                         from_tensor_format(pred.data.cpu().numpy())[0],
    #                         from_tensor_format(im_und.numpy())[0],
    #                         from_tensor_format(mask.data.cpu().numpy(), mask=True)[0]))

    #         test_batches += 1
    #         # 记录测试损失
    #         writer.add_scalar('Loss/Test', err.item(), epoch * len(test) + test_batches)
    #         # if args.debug and test_batches == 20:
    #         if args.debug and test_batches == 50:
    #             break

    #     # 记录每个 epoch 的平均测试损失
    #     writer.add_scalar('Loss/Test_Avg', test_err / test_batches, epoch)
    #     # 记录每个 epoch 的 PSNR 值
    #     writer.add_scalar('PSNR/Base', base_psnr / (test_batches * batch_size), epoch)
    #     writer.add_scalar('PSNR/Test', test_psnr / (test_batches * batch_size), epoch)
        
    #     t_end = time.time()

    #     train_err /= train_batches
    #     validate_err /= validate_batches
    #     test_err /= test_batches
    #     base_psnr /= (test_batches*batch_size)
    #     test_psnr /= (test_batches*batch_size)

    #     # Then we print the results for this epoch:
    #     '''
    #     运行1个epoch打印如下：
    #     Epoch 1/1
    #     time: 13.080119132995605s
    #     training loss:         1156.800310
    #     validation loss:       0.010139
    #     test loss:             0.016800
    #     base PSNR:             20.242637
    #     test PSNR:             10.799398
    #     '''
    #     print("Epoch {}/{}".format(epoch+1, num_epoch))
    #     print(" time: {}s".format(t_end - t_start))
    #     print(" training loss:\t\t{:.6f}".format(train_err))
    #     print(" validation loss:\t{:.6f}".format(validate_err))
    #     print(" test loss:\t\t{:.6f}".format(test_err))
    #     print(" base PSNR:\t\t{:.6f}".format(base_psnr))
    #     print(" test PSNR:\t\t{:.6f}".format(test_psnr))
        
    #     # save the model
    #     if epoch in [1, 2, num_epoch-1]:
    #         if save_fig:

    #             for im_i, pred_i, und_i, mask_i in vis:
    #                 # print('im_i---------')
    #                 im = abs(np.concatenate([und_i[0], pred_i[0], im_i[0], im_i[0] - pred_i[0]], 1))
    #                 plt.imsave(join(save_dir, 'im{0}_x.png'.format(i)), im, cmap='gray')

    #                 im = abs(np.concatenate([und_i[..., 0], pred_i[..., 0],
    #                                          im_i[..., 0], im_i[..., 0] - pred_i[..., 0]], 0))
    #                 plt.imsave(join(save_dir, 'im{0}_t.png'.format(i)), im, cmap='gray')
    #                 plt.imsave(join(save_dir, 'mask{0}.png'.format(i)),
    #                            np.fft.fftshift(mask_i[..., 0]), cmap='gray')
    #                 i += 1

    #         name = '%s_epoch_%d.npz' % (model_name, epoch)
    #         torch.save(rec_net.state_dict(), join(save_dir, name))
    #         print('model parameters saved at %s' % join(os.getcwd(), name))
    #         # print('')
    # # 训练结束后关闭 SummaryWriter
    # writer.close()
