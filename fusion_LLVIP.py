from utils.H5_read import H5ImageTextDataset, BinaryDataset_withText_Mask
from utils.img_read_save import img_save
from net.FDFusion import Net
import math
import os
import sys
import warnings
import cv2
import numpy as np
import sklearn.metrics as skm
import torch
from scipy.signal import convolve2d
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import argparse


sys.path.append(os.getcwd())
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings('ignore')  # 不显示warnings
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


parser = argparse.ArgumentParser(description="Test FDFusion on LLVIP dataset")
parser.add_argument('--model_path', type=str, default='07-09-11-41_lr_0.0001_module__batch_1', help='Path to the pre-trained model')
parser.add_argument('--dataset_name', type=str, default='LLVIP', help='Name of the dataset')
parser.add_argument('--pth_epoch', type=str, default='70', help='Checkpoint epoch to load')
args = parser.parse_args()
model_path = args.model_path
dataset_name = args.dataset_name
pth_epoch = args.pth_epoch

with open(os.path.join(r"./VLFDataset_h5", dataset_name + '_split.json'), "r") as f:
    splits = json.load(f)

testloader = DataLoader(
    BinaryDataset_withText_Mask(os.path.join(r"./VLFDataset_h5", dataset_name + '_VI+IR+Mask+text.h5'),
                                keys=splits['test'], dataset=dataset_name),
    batch_size=1,
    shuffle=True,
    drop_last=True,
    num_workers=0,
)


ckpt_path = os.path.join(r'./exp_LLVIP', model_path, "model", "ckpt_" + pth_epoch + '.pth')
save_path = os.path.join(r"./output", dataset_name, model_path, "ckpt_" + pth_epoch)
os.makedirs(save_path, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net(hidden_dim=256).to(device)


model.load_state_dict(torch.load(ckpt_path)['model'])
model.eval()

with torch.no_grad():
    for data_IR, data_VIS, textA, textB, Mask, index in tqdm(testloader):
        textA = textA.squeeze(1).cuda()
        textB = textB.squeeze(1).cuda()
        # data_IR = torch.FloatTensor(data_IR)
        # data_VIS = torch.FloatTensor(data_VIS)
        data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
        data_Fuse = model(data_IR, data_VIS, textA, textB)[0]
        data_Fuse = (data_Fuse - torch.min(data_Fuse)) / (torch.max(data_Fuse) - torch.min(data_Fuse))
        fi = np.squeeze((data_Fuse * 255).detach().cpu().numpy())
        fi = fi.astype('uint8')
        img_save(fi, index[0], save_path)



