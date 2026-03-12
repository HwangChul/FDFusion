import os
from utils.img_read_save import image_read_cv2
import h5py
import numpy as np
from tqdm import tqdm
import cv2
import torch
import pickle


img_text_path = 'VLFDataset'
h5_path = "VLFDataset_h5"
os.makedirs(h5_path, exist_ok=True)
task_name = 'IVF'
dataset_name = 'MSRS'
dataset_mode = 'train'
size = 'small'
mode = 'GRAY'
h5_file_path = os.path.join(h5_path, dataset_name + '' + dataset_mode +'.h5')
small_size_0 = (384, 288)
small_size_1 = (288, 384)


with h5py.File(h5_file_path, 'w') as h5_file:
    imageA = h5_file.create_group('imageA')     # IR
    imageB = h5_file.create_group('imageB')     # VI
    textA = h5_file.create_group('textA')
    textB = h5_file.create_group('textB')


    text_txt = os.path.join(r'D:\Edge_Downloads\VLFDataset\Image\IVF', dataset_name, dataset_mode + '.txt')
    with open(text_txt, 'r') as file:
        file_list = [line.strip() for line in file.readlines()]
    sample_names = []
    for name in file_list:
        name = name.split('.')[0]
        sample_names.append(name)

    if task_name == 'IVF':
        IR_all_token = torch.load(os.path.join('SmolVLM', 'tokens', 'fine_tune', dataset_name, 'IR_' + dataset_mode + '.pt'))
        VI_all_token = torch.load(os.path.join('SmolVLM', 'tokens', 'fine_tune', dataset_name, 'VI_' + dataset_mode + '.pt'))

        for sample_name in tqdm(sample_names):
            if size == 'small':
                img_A = image_read_cv2(os.path.join(img_text_path, 'Image', task_name, dataset_name, 'IR', sample_name + '.png'), mode='GRAY')
                h, w = img_A.shape
                if h < w:
                    img_A = cv2.resize(img_A, small_size_0)[None, ...] / 255.0
                    if mode == 'RGB':
                        img_B = cv2.resize(image_read_cv2(os.path.join(img_text_path, 'Image', task_name, dataset_name, 'VI', sample_name + '.png'), mode=mode), small_size_0) / 255.0
                        img_B = img_B.transpose(2, 0, 1)
                    else:
                        img_B = cv2.resize(image_read_cv2(os.path.join(img_text_path, 'Image', task_name, dataset_name, 'VI', sample_name + '.png'), mode='GRAY'), small_size_0)[None, ...] / 255.0
                else:
                    img_A = cv2.resize(img_A, small_size_1).T[None, ...] / 255.0
                    if mode == 'RGB':
                        img_B = cv2.resize(image_read_cv2(os.path.join(img_text_path, 'Image', task_name, dataset_name, 'VI', sample_name + '.png'), mode=mode), small_size_1).T / 255.0
                        img_B = img_B.transpose(2, 0, 1)
                    else:
                        img_B = cv2.resize(image_read_cv2(os.path.join(img_text_path, 'Image', task_name, dataset_name, 'VI', sample_name + '.png'), mode='GRAY'), small_size_1).T[None, ...] / 255.0

            else:
                img_A = image_read_cv2(os.path.join(img_text_path, 'Image', task_name, dataset_name, 'IR', sample_name + '.png'), mode='GRAY')[None, ...] / 255.0
                if mode == 'RGB':
                    img_B = image_read_cv2(os.path.join(img_text_path, 'Image', task_name, dataset_name, 'VI', sample_name + '.png'), mode=mode) / 255.0
                    img_B = img_B.transpose(2, 0, 1)
                else:
                    img_B = image_read_cv2(os.path.join(img_text_path, 'Image', task_name, dataset_name, 'VI', sample_name + '.png'), mode='GRAY')[None, ...] / 255.0

            IR_token = IR_all_token[sample_name+'.png']
            VI_token = VI_all_token[sample_name+'.png']
            IR_token_numpy = IR_token.unsqueeze(0).detach().to(torch.float32).numpy()
            VI_token_numpy = VI_token.unsqueeze(0).detach().to(torch.float32).numpy()

            imageA.create_dataset(sample_name, data=img_A)
            imageB.create_dataset(sample_name, data=img_B)
            textA.create_dataset(sample_name, data=IR_token_numpy)
            textB.create_dataset(sample_name, data=VI_token_numpy)


