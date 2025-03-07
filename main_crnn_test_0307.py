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


from cascadenet_pytorch.model_pytorch import *
from cascadenet_pytorch.dnn_io import to_tensor_format
from cascadenet_pytorch.dnn_io import from_tensor_format
# 导入 TensorBoard 模块
from torch.utils.tensorboard import SummaryWriter

# # PyTorch建议在使用多线程时设置OMP_NUM_THREADS环境变量，以避免系统过载。
# os.environ['OMP_NUM_THREADS'] = '1'
# # 设置PYTORCH_CUDA_ALLOC_CONF环境变量，以减少CUDA内存碎片
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
# # os.environ["CUDA_VISIBLE_DEVICES"] = "3" #,0,1,2,4,5,6,7
# # os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 指定使用 GPU 1 和 GPU 4
# # os.environ['CUDA_VISIBLE_DEVICES'] = '6'  # 指定使用 GPU 1 和 GPU 4
# # os.environ['CUDA_VISIBLE_DEVICES'] = '4'  # 指定使用 GPU 1 和 GPU 4
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'  # 指定使用 GPU 1 和 GPU 4

# # 设置环境变量 CUDA_VISIBLE_DEVICES  0-5(nvidia--os) 2-6 3-7
# # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 指定使用 GPU 1 和 GPU 4
# # os.environ['CUDA_VISIBLE_DEVICES'] = '4,7'  # 指定使用 GPU 7 和 GPU 3
# # os.environ['CUDA_VISIBLE_DEVICES'] = '1,4'  # 指定使用 GPU 4 和 GPU 7
# # os.environ['CUDA_VISIBLE_DEVICES'] = '1,4'  # 指定使用 GPU 4 和 GPU 7
# # os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'  # 指定使用 GPU 4 和 GPU 6
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# PyTorch建议在使用多线程时设置OMP_NUM_THREADS环境变量，以避免系统过载。
os.environ['OMP_NUM_THREADS'] = '1'
# 设置PYTORCH_CUDA_ALLOC_CONF环境变量，以减少CUDA内存碎片
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
# The above code is setting an environment variable `PYTORCH_CUDA_ALLOC_CONF` to the value
# `'max_split_size_mb:256'`. This environment variable is likely being used to configure memory
# allocation settings for PyTorch when running on a CUDA-enabled GPU. In this specific case, it
# appears to be setting the maximum split size for memory allocation to 256 megabytes.
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
# os.environ["CUDA_VISIBLE_DEVICES"] = "3" #,0,1,2,4,5,6,7
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 指定使用 GPU 1 和 GPU 4
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'  # 指定使用 GPU 1 和 GPU 4
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 指定使用 GPU 1 和 GPU 4

def prep_input(im, acc=4.0,centred=False):
    """Undersample the batch, then reformat them into what the network accepts.

    Parameters
    ----------
    gauss_ivar: float - controls the undersampling rate.
                        higher the value, more undersampling
    """
    # mask = cs.cartesian_mask(im.shape, acc, sample_n=8)
    # mask = cs.cartesian_mask(im.shape, acc, sample_n=8,centred=centred)
    # print('prep_input-centred',centred)
    acc = int(acc)
    #尝试另外一个mask
    mask = cs.shear_grid_mask(im.shape[1:], acc, sample_low_freq=True, sample_n=4)
    mask = np.repeat(mask[np.newaxis], im.shape[0], axis=0)
    
    # prep_input-mask-dtype: float64
    # print('prep_input-mask-dtype:',mask.dtype)
    # print('prep_input-im-shape:',im.shape) #prep_input-im-shape: (1, 30, 256, 256)
    # print('prep_input-mask-shape:',mask.shape)# prep_input-mask-shape: (1, 30, 256, 256)
    # im_und, k_und = cs.undersample(im, mask, centred=False, norm='ortho')
    # im_und, k_und = cs.undersample(im, mask, centred=centred, norm='ortho')
    
    # im_und = im_und / np.max(np.abs(im_und))  # 添加归一化
    # 将kspace中心化
    # im_und, k_und = cs.undersample(im, mask, centred=True, norm='ortho')
    
    # 动态归一化：基于当前样本的复数幅度最大值
    max_magnitude = np.max(np.abs(im))  # 获取幅度最大值
    max_magnitude = max(max_magnitude, 1e-3)  # 防止过小值
    im_normalized = im.astype(np.complex64) / (max_magnitude + 1e-8)  # 保持复数类型
    # 生成下采样数据
    im_und, k_und = cs.undersample(im_normalized, mask, centred=centred, norm='ortho')
    
    # 将数据转换为 torch.float32 类型，减少内存占用
    im_gnd_l = torch.from_numpy(to_tensor_format(im)).float()
    im_und_l = torch.from_numpy(to_tensor_format(im_und)).float()
    k_und_l = torch.from_numpy(to_tensor_format(k_und)).float()
    mask_l = torch.from_numpy(to_tensor_format(mask, mask=True)).float()
    # prep_input-mask_l-shape: torch.Size([4, 2, 256, 32, 30])
    # prep_input-mask_l-dtype: torch.float64
    # print('prep_input-mask_l-shape:',mask_l.shape)
    # print('prep_input-mask_l-dtype:',mask_l.dtype)
    # prep_input-im_und_l-shape: torch.Size([1, 2, 256, 256, 30])
    # prep_input-k_und_l-shape: torch.Size([1, 2, 256, 256, 30])
    # print('prep_input-im_und_l-shape:',im_und_l.shape)
    # print('prep_input-k_und_l-shape:',k_und_l.shape)
    print(f"[DEBUG] 输入范围: im_und={torch.min(im_und_l):.4f}~{torch.max(im_und_l):.4f}, im_gnd={torch.min(im_gnd_l):.4f}~{torch.max(im_gnd_l):.4f}")
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
    nx, ny, nt = data.shape  # 原始数据形状:  [256, 256, 30]
    
    # 转置数据，将时间维度放在最前面
    data_t = np.transpose(data, (2, 0, 1))  # 转置后形状: [30, 256, 256]
    
    # 生成训练集、验证集和测试集
    # 保持后两个维度为 [256, 256]，仅从时间维度切片
    train = np.array([data_t for _ in range(20)])  # 训练集: [20, 30, 256, 256]
    validate = np.array([data_t for _ in range(2)])  # 验证集: [2, 30, 256, 256]
    test = np.array([data_t for _ in range(2)])  # 测试集: [2, 30, 256, 256]

    return train, validate, test



# nohup python main_crnn_test.py --acceleration_factor 4 > output_0112_2.log 2>&1 &
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', metavar='int', nargs=1, default=['1'],
                        help='number of epochs')
    # parser.add_argument('--num_epoch', metavar='int', nargs=1, default=['300'],
    #                     help='number of epochs')
    # parser.add_argument('--num_epoch', metavar='int', nargs=1, default=['400'],
    #                     help='number of epochs')
    # parser.add_argument('--batch_size', metavar='int', nargs=1, default=['1'],
    #                     help='batch size')
    parser.add_argument('--batch_size', metavar='int', nargs=1, default=['1'],
                        help='batch size')
    parser.add_argument('--lr', metavar='float', nargs=1,
                        default=['0.000001'], help='initial learning rate')
    # parser.add_argument('--lr', metavar='float', nargs=1,
    #                     default=['0.00005'], help='initial learning rate')
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
    model_name = 'crnn_mri_0307_kspace_center_1'
    # 模型中间结果保存位置
    model_save_dir = './saved_data/0307_1'
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
    # data_shape = [Nx, Ny, Nt]
    # Test creating mask and compute the acceleration rate
    # dummy_mask = cs.cartesian_mask(data_shape, acc, sample_n=8)
    # dummy_mask = cs.cartesian_mask((10, Nx, Ny//Ny_red), acc, sample_n=8)
    # sample_und_factor = cs.undersampling_rate(dummy_mask)
    # # Undersampling Rate: 0.25
    # print('Undersampling Rate: {:.2f}'.format(sample_und_factor))

    # Specify network
    # rec_net = CRNN_MRI()
    # 初始化模型和优化器
    rec_net = CRNN_MRI().cuda()
    # Parameter Count: 297794
    print("Parameter Count: %d" % count_parameters(rec_net))
    
    # 添加梯度裁剪（PyTorch示例）
    torch.nn.utils.clip_grad_norm_(rec_net.parameters(), max_norm=1.0)
    # torch.nn.utils.clip_grad_norm_(rec_net.parameters(), max_norm=5.0)
    
    # 确保模型参数是 FP32
    rec_net.float()

    criterion = torch.nn.MSELoss()
    #使用混合精度训练
    # scaler = GradScaler()
    # 修改后 (添加自动缩放策略)
    scaler = GradScaler(init_scale=1024, growth_interval=200)
    # scaler = torch.amp.GradScaler('cuda')
    # scaler = torch.amp.GradScaler()

    
    # # build CRNN-MRI with pre-trained parameters
    # rec_net.load_state_dict(torch.load('./models/pretrained/crnn_mri_d5_c5.pth'))

    if cuda:
        rec_net = rec_net.cuda()
        criterion.cuda()

    centred = False
    i = 0
    # 定义梯度累积的步数
    # accumulation_steps = 4  # 可以根据显存大小调整
    # optimizer = optim.Adam(rec_net.parameters(), lr=float(args.lr[0]) * accumulation_steps, betas=(0.5, 0.999))
    optimizer = optim.Adam(rec_net.parameters(), lr=float(args.lr[0]) * accumulation_steps, betas=(0.9, 0.999))
    for epoch in range(num_epoch):
        t_start = time.time()
        # Training
        train_err = 0
        train_batches = 0
        # 训练
        rec_net.train()
        # 获取总 batch 数
        total_batches_train = len(train) // batch_size
        total_batches_validate = len(validate) // batch_size
        total_batches_test = len(test) // batch_size
               
        for batch_idx, im in enumerate(iterate_minibatch(train, batch_size, shuffle=True)):
            # 检查是否是最后一个 epoch 的最后一个 batch
            is_last_epoch = (epoch == num_epoch - 1)
            is_last_batch = (batch_idx == total_batches_train - 1)
            save_last = is_last_epoch and is_last_batch

            # 数据预处理
            # im_undersample, k_undersample, mask, im_groudtruth = prep_input(im, acc,centred=False)
            im_undersample, k_undersample, mask, im_groudtruth = prep_input(im, acc,centred=centred)
            # 转换为 PyTorch 变量
            im_undersample = Variable(im_undersample.type(Tensor))
            k_undersample = Variable(k_undersample.type(Tensor))
            mask = Variable(mask.type(Tensor))
            groudtruth = Variable(im_groudtruth.type(Tensor))
            # print('im_undersample-shape:',im_undersample.shape) #torch.Size([1, 2, 256, 256, 30])
            # print('im_undersample-dtype:',im_undersample.dtype) #torch.float32
            # print('k_undersample-shape:',k_undersample.shape) #torch.Size([1, 2, 256, 256, 30])
            # print('k_undersample-dtype:',k_undersample.dtype) #torch.float32
            # print('mask-shape:',mask.shape) #torch.Size([1, 2, 256, 256, 30])
            # print('mask-dtype:',mask.dtype) #float32
            # print('groudtruth-shape:',groudtruth.shape) #torch.Size([1, 2, 256, 256, 30])
            # print('groudtruth-dtype:',groudtruth.dtype) #torch.float32

            # 前向传播
            optimizer.zero_grad()
            with autocast('cuda'):
                rec = rec_net(im_undersample, k_undersample, mask, test=False)
                loss = criterion(rec, groudtruth)

            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 记录训练损失
            train_err += loss.item()
            train_batches += 1
            writer.add_scalar('Loss/Train', loss.item(), epoch * len(train) + train_batches)

            # 保存最后一个 epoch 和最后一个 batch 的数据
            if save_last:
                # 将 Tensor 转换为 numpy 数组
                # 调整维度，让最后一个维度的大小为 2
                im_undersample_permuted = im_undersample.permute(0, 4, 2, 3, 1)
                # print('调整后 im_undersample 形状:', im_undersample_permuted.shape) #调整后 im_undersample 形状: torch.Size([1, 30, 256, 256, 2])
                im_undersample_complex  = torch.view_as_complex(im_undersample_permuted.contiguous())
                im_undersample_np = im_undersample_complex.cpu().numpy()
                # print('im_undersample_np-shape:',im_undersample_np.shape) #im_undersample_np-shape: (1, 30, 256, 256)
                k_undersample_np = k_undersample.cpu().numpy()
                # print('k_undersample_np-shape:',k_undersample_np.shape) #k_undersample_np-shape: (1, 2, 256, 256, 30)
                mask_np = mask.cpu().numpy()
                groudtruth_np = groudtruth.cpu().numpy()
                
                # print('train-im_undersample-shape:',im_undersample_np.shape) #train-im_undersample-shape: (1, 2, 256, 256, 30)
                # print('train-k_undersample-shape:',k_undersample_np.shape) #train-k_undersample-shape: (1, 2, 256, 256, 30)
                # print('train-mask-shape:',mask_np.shape) #train-mask-shape: (1, 2, 256, 256, 30)
                # print('train-im_groudtruth-shape:',groudtruth_np.shape) #train-im_groudtruth-shape: (1, 2, 256, 256, 30)

                # 创建 train_output 子目录
                train_output_dir = join(save_dir, 'train_output')
                os.makedirs(train_output_dir, exist_ok=True)  # 如果目录不存在则创建

                # 保存为 .mat 格式
                savemat(join(train_output_dir, 'im_undersample.mat'), {'im_undersample': im_undersample_np})
                savemat(join(train_output_dir, 'k_undersample.mat'), {'k_undersample': k_undersample_np})
                savemat(join(train_output_dir, 'mask.mat'), {'mask': mask_np})
                savemat(join(train_output_dir, 'groudtruth.mat'), {'groudtruth': groudtruth_np})

                # 保存为 .npy 格式
                np.save(join(train_output_dir, 'im_undersample.npy'), im_undersample_np)
                np.save(join(train_output_dir, 'k_undersample.npy'), k_undersample_np)
                np.save(join(train_output_dir, 'mask.npy'), mask_np)
                np.save(join(train_output_dir, 'groudtruth.npy'), groudtruth_np)

                # 保存为 .png 格式
                # plt.imsave(join(train_output_dir, 'im_undersample.png'), np.abs(im_undersample_np[0, 0, :, :, 0]), cmap='gray')
                plt.imsave(join(train_output_dir, 'im_undersample.png'), np.abs(im_undersample_np[0, 0, :, :]), cmap='gray')
                plt.imsave(join(train_output_dir, 'mask.png'), np.abs(mask_np[0, 0, :, :, 0]), cmap='gray')
                plt.imsave(join(train_output_dir, 'groudtruth.png'), np.abs(groudtruth_np[0, 0, :, :, 0]), cmap='gray')

                # 将 k-space 数据转换到图像域并保存
                k_undersample_complex = k_undersample_np[0, 0, :, :, 0] + 1j * k_undersample_np[0, 1, :, :, 0]
                image_from_k_space = np.fft.ifft2(k_undersample_complex)
                image_from_k_space = np.abs(image_from_k_space)
                plt.imsave(join(train_output_dir, 'image_from_k_space.png'), image_from_k_space, cmap='gray')

                print(f"Saved im_undersample, k_undersample, mask, and groudtruth to {train_output_dir}")
            # 释放显存
            torch.cuda.empty_cache()
            del im_undersample, k_undersample, mask, groudtruth, rec, loss

            if args.debug and train_batches == 50:
                break

        # 记录每个 epoch 的平均训练损失
        writer.add_scalar('Loss/Train_Avg', train_err / train_batches, epoch)
        
        validate_err = 0
        validate_batches = 0
        rec_net.eval()
        # for im in iterate_minibatch(validate, batch_size, shuffle=False):
        for batch_idx, im in enumerate(iterate_minibatch(validate, batch_size, shuffle=False)):
            # im_und, k_und, mask, im_gnd = prep_input(im, acc)
            # im_und, k_und, mask, im_gnd = prep_input(im, acc,centred=False)
            im_und, k_und, mask, im_gnd = prep_input(im, acc,centred=centred)
            # 检查是否是最后一个 epoch 的最后一个 batch
            is_last_epoch = (epoch == num_epoch - 1)
            is_last_batch = (batch_idx == total_batches_validate - 1)
            save_last = is_last_epoch and is_last_batch
            # print('iterate_minibatch-validate-save_last:',save_last)
            
            with torch.no_grad():
                im_u = Variable(im_und.type(Tensor))
                k_u = Variable(k_und.type(Tensor))
                mask = Variable(mask.type(Tensor))
                gnd = Variable(im_gnd.type(Tensor))
                pred = rec_net(im_u, k_u, mask, test=True)
            # torch.cuda.empty_cache()
            # 
            torch.cuda.empty_cache()
            err = criterion(pred, gnd)

            validate_err += err
            validate_batches += 1
            # 记录验证损失
            writer.add_scalar('Loss/Validate', err.item(), epoch * len(validate) + validate_batches)


            if args.debug and validate_batches == 20:
                break
        # 记录每个 epoch 的平均验证损失
        writer.add_scalar('Loss/Validate_Avg', validate_err / validate_batches, epoch)
        
        vis = []
        test_err = 0
        base_psnr = 0
        test_psnr = 0
        test_batches = 0
        # for im in iterate_minibatch(test, batch_size, shuffle=False):
        for batch_idx, im in enumerate(iterate_minibatch(test, batch_size, shuffle=True)):
            # im_und, k_und, mask, im_gnd = prep_input(im, acc)
            # im_und, k_und, mask, im_gnd = prep_input(im, acc,centred=False)
            im_und, k_und, mask, im_gnd = prep_input(im, acc,centred=centred)
            with torch.no_grad():
                im_u = Variable(im_und.type(Tensor))
                k_u = Variable(k_und.type(Tensor))
                mask = Variable(mask.type(Tensor))
                gnd = Variable(im_gnd.type(Tensor))
                # pred = rec_net(im_u, k_u, mask, test=True)
                # 检查是否是最后一个 epoch 的最后一个 batch
                is_last_epoch = (epoch == num_epoch - 1)
                is_last_batch = (batch_idx == total_batches_test - 1)
                save_last = is_last_epoch and is_last_batch
                # print('iterate_minibatch-test-save_last:',save_last)
                # pred = rec_net(im_u, k_u, mask, test=True)
                pred = rec_net(im_u, k_u, mask, test=True, model_save_dir =model_save_dir ,save_last=save_last)
                print('pred-shape:',pred.shape) #torch.Size([1, 2, 256, 256, 30])
            # torch.cuda.empty_cache()
            
            # 
            torch.cuda.empty_cache()
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
            # 记录测试损失
            writer.add_scalar('Loss/Test', err.item(), epoch * len(test) + test_batches)
            # if args.debug and test_batches == 20:
            if args.debug and test_batches == 50:
                break

        # 记录每个 epoch 的平均测试损失
        writer.add_scalar('Loss/Test_Avg', test_err / test_batches, epoch)
        # 记录每个 epoch 的 PSNR 值
        writer.add_scalar('PSNR/Base', base_psnr / (test_batches * batch_size), epoch)
        writer.add_scalar('PSNR/Test', test_psnr / (test_batches * batch_size), epoch)
        
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
        
        print(" base PSNR max:\t\t{:.6f}".format(np.max(base_psnr)))
        print(" test PSNR max:\t\t{:.6f}".format(np.max(test_psnr)))
        print(" base PSNR min:\t\t{:.6f}".format(np.min(base_psnr)))
        print(" test PSNR min:\t\t{:.6f}".format(np.min(test_psnr)))
        
        # save the model
        if epoch in [1, 2, num_epoch-1]:
            if save_fig:

                for im_i, pred_i, und_i, mask_i in vis:
                    # print('im_i---------')
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
            # print('')
    # 训练结束后关闭 SummaryWriter
    writer.close()