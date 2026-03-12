# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''
import os
import matplotlib.pyplot as plt
import sys
import time
import datetime
import torch
from utils.Logger import Logger1
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from utils.lossfun import Fusionloss, LpLssimLossweight
import numpy as np
from utils.H5_read import H5ImageTextDataset
import argparse
import warnings
from net.FDFusion import Net
import logging
import shutil
import random

'''
------------------------------------------------------------------------------
'''


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(3407)

'''
------------------------------------------------------------------------------
'''
warnings.filterwarnings('ignore')

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
sys.path.append(os.getcwd())
logging.basicConfig(level=logging.CRITICAL)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

parser = argparse.ArgumentParser()
parser.add_argument('--i2t_dim', type=int, default=32, help='')
parser.add_argument('--hidden_dim', type=int, default=256, help='')
parser.add_argument('--numepochs', type=int, default=70, help='')
parser.add_argument('--lr', type=float, default=0.0001, help='')
parser.add_argument('--gamma', type=float, default=0.7, help='')
parser.add_argument('--step_size', type=int, default=6, help='')
parser.add_argument('--batch_size', type=int, default=1, help='')
parser.add_argument('--loss_grad_weight', type=int, default=20, help='')
parser.add_argument('--loss_int_weight', type=int, default=1, help='')
parser.add_argument('--loss_ssim', type=int, default=0, help='')
parser.add_argument('--dataset_path', type=str, default="./VLFDataset_h5/MSRS_train_IR&VI_token_.h5", help='')
parser.add_argument('--pre_train_path', type=str,
                    default=r"",
                    help='pre_train_path')
opt = parser.parse_args()

'''
------------------------------------------------------------------------------
Set the hyper-parameters for training
------------------------------------------------------------------------------
'''
pre_model = opt.pre_train_path
num_epochs = opt.numepochs
lr = opt.lr
step_size = opt.step_size
gamma = opt.gamma
weight_decay = 0
batch_size = opt.batch_size
weight_ingrad = opt.loss_grad_weight
weight_int = opt.loss_int_weight
weight_ssim = opt.loss_ssim
hidden_dim = opt.hidden_dim
i2t_dim = opt.i2t_dim
dataset_path = opt.dataset_path
module = ""
alpha = 1

'''
------------------------------------------------------------------------------
model
------------------------------------------------------------------------------
'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net(hidden_dim=hidden_dim, image2text_dim=i2t_dim)
model.to(device)
criterion = LpLssimLossweight().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

trainloader = DataLoader(H5ImageTextDataset(dataset_path), batch_size=batch_size,
                         shuffle=True, num_workers=0, drop_last=True)
time_begin = time.strftime("%y_%m_%d_%H_%M", time.localtime())
save_path = "exp/" + str(time_begin) + '_lr_%s' % (str(opt.lr)) + '_module_%s' % (
        str(module) + '_batch_%s' % (str(batch_size)) + '_step_%s' % (str(step_size)))
logger = Logger1(rootpath=save_path, timestamp=False)
params = {
    'epoch': num_epochs,
    'lr': lr,
    'batch_size': batch_size,
    'optim_step': step_size,
    'optim_gamma': gamma,
    'gradweight': weight_ingrad,
    'weight_decay': weight_decay,
}
logger.save_param(params)
logger.new_subfolder('model')
writer = SummaryWriter(logger.logpath)
destination_folder = os.path.join(save_path, 'code')



def save_code_files(source_file, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    dest_train_file_path = os.path.join(destination_folder, os.path.basename(source_file))
    shutil.copyfile(source_file, dest_train_file_path)
    net_src = 'net'
    net_dst = os.path.join(destination_folder, 'net')
    if os.path.exists(net_dst):
        shutil.rmtree(net_dst)
    shutil.copytree(net_src, net_dst)

save_code_files(os.path.basename(__file__), destination_folder)
'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''
start_epoch, step = 0, 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()
start_time = time.time()
loss = Fusionloss(coeff_int=weight_int, coeff_grad=weight_ingrad, device=device)

# 继续训练
if pre_model != "":
    checkpoint_path = os.path.join('exp', pre_model, 'model', 'ckpt_50.pth')
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer1'])
    scheduler.load_state_dict(checkpoint['lr_schedule1'])
    print('load_pretrain_model')

    start_epoch = checkpoint['epoch'] + 1
    step = checkpoint['step']

for epoch in range(start_epoch, num_epochs):
    ''' train '''
    s_temp = time.time()
    model.train()
    loss_total = 0
    loss_in_grad_total = 0

    lpA_total = 0
    lpB_total = 0
    lssimA_total = 0
    lssimB_total = 0
    lplssimA_total = 0
    lplssimB_total = 0
    for i, (data_IR, data_VI, textA, textB, index) in enumerate(trainloader):
        data_VI, data_IR = data_VI.to(device), data_IR.to(device)
        textA = textA.squeeze(1).to(device)
        textB = textB.squeeze(1).to(device)
        F = model(data_IR, data_VI, textA, textB)
        batchsize, channels, rows, columns = data_IR.shape
        weighttemp = int(np.sqrt(rows * columns))
        lplssimA, lpA, lssimA = criterion(image_in=data_IR, image_out=F, weight=weighttemp)
        lplssimB, lpB, lssimB = criterion(image_in=data_VI, image_out=F, weight=weighttemp)
        loss_in_grad, _, _ = loss(data_IR, data_VI, F)
        loss_ssim = lplssimA + alpha * lplssimB
        lossALL = loss_in_grad + weight_ssim * loss_ssim
        optimizer.zero_grad()
        lossALL.backward()
        optimizer.step()

        loss_total += lossALL.item()
        loss_in_grad_total += loss_in_grad.item()

        lpA_total += lpA.item()
        lpB_total += lpB.item()
        lssimA_total += lssimA.item()
        lssimB_total += lssimB.item()
        lplssimA_total += lplssimA.item()
        lplssimB_total += lplssimB.item()

        batches_done = epoch * len(trainloader) + i
        batches_left = num_epochs * len(trainloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))

        # 打印当前的训练进度信息
        logger.log_and_print(
            "[Epoch %d/%d] [Batch %d/%d] [loss: %.4f] [loss_in_grad: %.4f] [lplssimA: %.4f] [lplssimB: %.4f] [lpA: %.4f] [lssimA: %.4f] [lpB: %.4f] [lssimB: %.4f] ETA: %.10s epoch/s: %ds"
            % (
                epoch + 1,
                num_epochs,
                i,
                len(trainloader),
                lossALL.item(),
                loss_in_grad.item(),
                lplssimA.item(),
                lplssimB.item(),
                lpA.item(),
                lssimA.item(),
                lpB.item(),
                lssimB.item(),
                time_left,
                time.time() - prev_time,
            )
        )

        prev_time = time.time()


        writer.add_scalar('loss/01 Loss', lossALL.item(), step)
        writer.add_scalar('loss/01 loss_in_grad', loss_in_grad.item(), step)
        writer.add_scalar('loss/01 lplssimA', lplssimA.item(), step)
        writer.add_scalar('loss/01 lplssimB', lplssimB.item(), step)
        writer.add_scalar('loss/14 learning rate', optimizer.state_dict()['param_groups'][0]['lr'], step)
        step += 1

        if (epoch + 1) % 1 == 0:
            if i < 20:
                for j in range(data_IR.shape[0]):

                    temp = np.zeros((rows, 3 * columns))

                    temp[:rows, 0:columns] = np.squeeze(data_IR[j].detach().cpu().numpy()) * 255
                    temp[:rows, columns:columns * 2] = np.squeeze(data_VI[j].detach().cpu().numpy()) * 255
                    temp[:rows, columns * 2:columns * 3] = np.squeeze(F[j].detach().cpu().numpy()) * 255

                    if not os.path.exists(os.path.join(logger.logpath, 'pic_fusion', "ckpt_" + str(epoch + 1))):
                        os.makedirs(os.path.join(logger.logpath, 'pic_fusion', "ckpt_" + str(epoch + 1)))
                    plt.imsave(os.path.join(logger.logpath, 'pic_fusion', "ckpt_" + str(epoch + 1),
                                            str(index[j]) + '.png'),
                               temp,
                               cmap="gray")

    scheduler.step()

    if (epoch + 1) % 1 == 0:
        checkpoint = {
            'model': model.state_dict(),
            'optimizer1': optimizer.state_dict(),
            'lr_schedule1': scheduler.state_dict(),
            "epoch": epoch,
            'step': step,
        }
        os.path.join(logger.logpath, 'model')
        torch.save(checkpoint, os.path.join(logger.logpath, 'model', 'ckpt_%s.pth' % (str(epoch + 1))))
    e_temp = time.time()
    print("This Epoch takes time: " + str(e_temp - s_temp))
    print(
        f"loss_total: {loss_total:.4f}, int_grad: {loss_in_grad_total:.4f}, "
        f"lplssimA: {lplssimA_total / len(trainloader):.4f}, "
        f"lplssimB: {lplssimB_total / len(trainloader):.4f} "
        f"lpA: {lpA_total / len(trainloader):.4f}, lssimA: {lssimA_total / len(trainloader):.4f}, "
        f"lpB: {lpB_total / len(trainloader):.4f}, lssimB: {lssimB_total / len(trainloader):.4f} ")
    log_file = os.path.join(logger.logpath, module + '_loss.txt')
    # 增加每个epoch总的loss
    with open(log_file, "a") as f:
        f.write(f"Epoch {epoch + 1}/{num_epochs}"
                f" - Loss Total: {loss_total:.4f}, "
                f"int_grad: {loss_in_grad_total:.4f}, "
                f"lplssimA: {lplssimA_total / len(trainloader):.4f}, "
                f"lplssimB: {lplssimB_total / len(trainloader):.4f}, "
                f"lpA: {lpA_total / len(trainloader):.4f}, "
                f"lssimA: {lssimA_total / len(trainloader):.4f}, "
                f"lpB: {lpB_total / len(trainloader):.4f}, "
                f"lssimB: {lssimB_total / len(trainloader):.4f}, "
                f"This Epoch takes time: {str(e_temp - s_temp)}\n\n")

end_time = time.time()
logger.log_and_print("total_time: " + str(end_time - start_time))


os.system("/usr/bin/shutdown")
