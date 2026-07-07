# !/user/bin/env python3
# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
"""
import datetime
import argparse
import json
import os
import random
import shutil
import time
import torch.nn.functional as F
import kornia
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from net.FDFusion import Net
from utils.H5_read import BinaryDataset_withText_Mask
from utils.Logger import Logger1
from utils.loss import Fusionloss, L_color, PixelwiseColorAngleLoss

"""
------------------------------------------------------------------------------
Environment setup and random seed for reproducibility
------------------------------------------------------------------------------
"""


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(3407)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
GPU_number = os.environ["CUDA_VISIBLE_DEVICES"]
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
"""
------------------------------------------------------------------------------
Training hyperparameters
------------------------------------------------------------------------------
"""
parser = argparse.ArgumentParser(description="Train FDFusion on LLVIP dataset")
parser.add_argument("--pre_train_path", type=str, default=r"")
parser.add_argument("--pre_train_epoch", type=int, default=None)
parser.add_argument("--mode", type=str, default="gray")
parser.add_argument("--ssim_wight", type=float, default=1)
parser.add_argument("--grad_wight", type=float, default=20)
parser.add_argument("--num_epochs", type=int, default=80)
parser.add_argument("--start_epoch", type=int, default=1)
parser.add_argument("--step", type=int, default=0)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--clip_grad_norm_value", type=float, default=0.01)
parser.add_argument("--optim_step", type=int, default=10)
parser.add_argument("--optim_gamma", type=float, default=0.7)
parser.add_argument("--module", type=str, default="")
args = parser.parse_args()

pre_train_path = args.pre_train_path
pre_train_epoch = args.pre_train_epoch
mode = args.mode
ssim_wight = args.ssim_wight
grad_wight = args.grad_wight
num_epochs = args.num_epochs
start_epoch = args.start_epoch
step = args.step
lr = args.lr
weight_decay = args.weight_decay
batch_size = args.batch_size
clip_grad_norm_value = args.clip_grad_norm_value
optim_step = args.optim_step
optim_gamma = args.optim_gamma
module = args.module
"""
------------------------------------------------------------------------------
Data loader
------------------------------------------------------------------------------
"""
dataset_name = "LLVIP"
with open(os.path.join(r"./VLFDataset_h5", dataset_name + '_split.json'), "r") as f:
    splits = json.load(f)
train_loader = DataLoader(
    BinaryDataset_withText_Mask(os.path.join(r"./VLFDataset_h5", dataset_name + '_VI+IR+Mask+text.h5'),
                                keys=splits['train'], dataset=dataset_name, mode=mode),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=0,
)

timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
if pre_train_path != "":
    save_path = "exp_LLVIP/" + pre_train_path
else:
    save_path = "exp_LLVIP/" + str(timestamp) + '_lr_%s' % (str(lr)) + '_module_%s' % (
            str(module) + '_batch_%s' % (str(batch_size)))
logger = Logger1(rootpath=save_path, timestamp=False)
logger.new_subfolder('model')
log_dir = os.path.join(r'/root/tf-logs/stage1/', f"{str(timestamp)}_{module}")
loss_logger = SummaryWriter(log_dir=log_dir)


def save_code_files(source_file, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    dest_train_file_path = os.path.join(destination_folder, os.path.basename(source_file))
    shutil.copyfile(source_file, dest_train_file_path)

    # 复制 net 文件夹下的所有文件
    net_src = 'net'
    net_dst = os.path.join(destination_folder, 'net')
    if os.path.exists(net_dst):
        shutil.rmtree(net_dst)
    shutil.copytree(net_src, net_dst)


save_code_files(os.path.basename(__file__), os.path.join(save_path, 'code'))
"""
------------------------------------------------------------------------------
Model initialization
------------------------------------------------------------------------------
"""
model = Net(hidden_dim=hidden_dim, image2text_dim=i2t_dim)


optimizer_stage1 = torch.optim.AdamW(list(encoder.parameters()) + list(decoder_stage1.parameters()),
                                    lr=lr, weight_decay=weight_decay)
# scheduler_stage1 = torch.optim.lr_scheduler.StepLR(optimizer_stage1, step_size=optim_step, gamma=optim_gamma)
scheduler_stage1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_stage1, T_max=num_epochs)

if pre_train_path != "":
    checkpoint_path = os.path.join('exp_LLVIP/', pre_train_path, 'model', f'ckpt_{str(pre_train_epoch)}.pth')
    checkpoint = torch.load(checkpoint_path)
    # 恢复模型、优化器和调度器的状态
    encoder.load_state_dict(checkpoint['encoder'], strict=True)
    decoder_stage1.load_state_dict(checkpoint['decoder_stage1'], strict=True)
    optimizer_stage1.load_state_dict(checkpoint['optimizer_stage1'])
    scheduler_stage1.load_state_dict(checkpoint['scheduler_stage1'])
    print('load_pretrain_model')

    # 恢复 epoch 和 step
    start_epoch = checkpoint['epoch'] + 1  # 下一个 epoch 从保存的 epoch + 1 开始
    step = checkpoint['step']  # 恢复 step
"""
------------------------------------------------------------------------------
Loss functions
------------------------------------------------------------------------------
"""
MSELoss = nn.MSELoss()
fusion_loss = Fusionloss(coeff_grad=grad_wight, device=device)
Loss_ssim = kornia.losses.SSIMLoss(11, reduction="mean")
cbcr_loss = L_color().to(device)

transform = transforms.Grayscale(num_output_channels=1)

"""
------------------------------------------------------------------------------
Training loop
------------------------------------------------------------------------------
"""
sample_count = len(train_loader)
prev_time = time.time()
start_time = time.time()
encoder.train()
decoder_stage1.train()
for epoch in range(start_epoch, num_epochs + 1):
    s_temp = time.time()
    lossALL_epoch = 0
    lossALL_ssim = 0
    loss_int_grad_epoch = 0
    for i, (image_IR, image_VIS, text_IR, text_VIS, mask, index) in enumerate(train_loader):
        image_IR, image_VIS = image_IR.to(device), image_VIS.to(device)         # FILM数据集训练不需要插值
        image_IR = F.interpolate(image_IR, size=[288, 384], mode='area')
        image_VIS = F.interpolate(image_VIS, size=[288, 384], mode='area')

        text_IR = text_IR.squeeze(1).to(device)
        text_VIS = text_VIS.squeeze(1).to(device)

        encoder.zero_grad()
        decoder_stage1.zero_grad()
        optimizer_stage1.zero_grad()

        fusion_image = model(image_IR, image_VIS, text_IR, text_VIS)

        loss_total, loss_in, loss_grad = fusion_loss(image_IR, image_VIS, fusion_image)

        IR_ssim_loss = Loss_ssim(fusion_image, image_IR)
        VIS_ssim_loss = Loss_ssim(fusion_image, image_VIS)
        ssim_loss = IR_ssim_loss + VIS_ssim_loss

        lossALL = loss_total + ssim_wight * ssim_loss
        lossALL.backward()

        global_step = (epoch - 1) * sample_count + i
        if global_step % 10 == 0:
            loss_logger.add_scalar('Train/Loss', lossALL.item(), global_step)

        lossALL_epoch += lossALL.item()
        loss_int_grad_epoch += loss_total.item()
        lossALL_ssim += ssim_loss.item()

        # 梯度裁剪
        nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(decoder_stage1.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        # 优化器更新
        optimizer_stage1.step()

        # 估计剩余时间并打印训练信息
        batches_done = epoch * sample_count + i
        batches_left = num_epochs * sample_count - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))

        # sys.stdout.write(
        logger.log_and_print(
            "\r[Epoch %d/%d] [Batch %d/%d] [lossALL: %.4f] [ssim: %.4f] [loss_in: %.4f] [loss_grad: %.4f] ETA: %.10s epoch/s: %ds"
            % (
                epoch,
                num_epochs,
                i,
                sample_count,
                lossALL.item(),
                ssim_loss.item(),
                loss_in.item(),
                loss_grad.item(),
                time_left,
                time.time() - prev_time,
            )
        )
        prev_time = time.time()
        batchsize, channels, rows, columns = image_IR.shape
        if epoch % 1 == 0 and i < 20:
            for j in range(image_IR.shape[0]):
                temp = np.zeros((rows, 3 * columns))

                temp[:rows, 0:columns] = np.squeeze(image_IR[j].detach().cpu().numpy()) * 255
                temp[:rows, columns:columns * 2] = np.squeeze(image_VIS[j].detach().cpu().numpy()) * 255
                temp[:rows, columns * 2:columns * 3] = np.squeeze(fusion_image[j].detach().cpu().numpy()) * 255
                save_dir = os.path.join(logger.logpath, 'pic_fusion', "ckpt_" + str(epoch))
                os.makedirs(save_dir, exist_ok=True)
                plt.imsave(os.path.join(save_dir, str(index[j]) + '.png'), temp, cmap="gray")

    avg_epoch_loss = lossALL_epoch / sample_count
    loss_logger.add_scalar('Train/Epoch_Loss', lossALL_epoch, epoch)
    loss_logger.add_scalar('Train/Epoch_int&grad_Loss', loss_int_grad_epoch, epoch)
    loss_logger.add_scalar('Train/Epoch_ssim_Loss', lossALL_ssim, epoch)

    # 学习率调整
    scheduler_stage1.step()
    # 学习率下限限制，防止过低
    if optimizer_stage1.param_groups[0]["lr"] <= 1e-6:
        optimizer_stage1.param_groups[0]["lr"] = 1e-6

    # 定期保存模型
    # if epoch > 15:
    if epoch % 5 == 0:
        checkpoint = {
            "encoder": encoder.state_dict(),
            "decoder_stage1": decoder_stage1.state_dict(),
            "optimizer_stage1": optimizer_stage1.state_dict(),
            "scheduler_stage1": scheduler_stage1.state_dict(),
            "epoch": epoch,
            'step': step,
        }
        torch.save(checkpoint, os.path.join(logger.logpath, 'model', 'ckpt_%s.pth' % (str(epoch))))
    e_temp = time.time()

    print(
        f"This Epoch takes time: {str(e_temp - s_temp)} seconds,"
        f"lossALL: {lossALL_epoch:.4f}")
    log_file = os.path.join(logger.logpath, module + '_loss.txt')
    # 增加每个epoch总的loss
    with open(log_file, "a") as f:
        f.write(f"Epoch {epoch}/{num_epochs}"
                f" - lossALL: {lossALL_epoch:.4f}, loss_ssim: {lossALL_ssim:.4f}, loss_int&grad: {loss_int_grad_epoch:.4f}"
                f"This Epoch takes time: {str(e_temp - s_temp)}\n\n")

end_time = time.time()
logger.log_and_print("total_time: " + str(end_time - start_time))

loss_logger.close()
os.system("/usr/bin/shutdown")
