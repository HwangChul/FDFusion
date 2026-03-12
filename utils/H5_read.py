import torch
from torch.utils.data import Dataset
import h5py
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class H5ImageTextDataset(Dataset):
    def __init__(self, h5_file_path):
        self.h5_file_path = h5_file_path

    def __len__(self):
        with h5py.File(self.h5_file_path, 'r') as h5_file:
            return len(h5_file['imageA'])

    def __getitem__(self, idx):
        with h5py.File(self.h5_file_path, 'r') as h5_file:
            group_names = list(h5_file.keys())
            sample_name = list(h5_file[group_names[0]].keys())[idx]
            imageA = torch.from_numpy(h5_file['imageA'][sample_name][()])
            imageB = torch.from_numpy(h5_file['imageB'][sample_name][()])
            textA = torch.from_numpy(h5_file['textA'][sample_name][()])
            textB = torch.from_numpy(h5_file['textB'][sample_name][()])
            return imageA, imageB, textA, textB, sample_name

if __name__ == '__main__':
    dataset_path = '../VLFDataset_h5/MSRS_train_IR&VI_depth.h5'
    trainloader = DataLoader(H5ImageTextDataset(dataset_path), batch_size=1,
                             shuffle=False, num_workers=0, drop_last=True)
    for i, (data_IR, data_VI, text_IR, text_VI, depth, index) in enumerate(trainloader):
        print(data_VI.shape)
        print(data_IR.shape)
        print(text_IR.shape)
        print(text_VI.shape)
        print(depth.shape)
