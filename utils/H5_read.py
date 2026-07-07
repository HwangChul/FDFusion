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


class BinaryDataset_withText_Mask(Dataset):
    def __init__(self, h5_file_path, keys=None, dataset="LLVIP", mode="gray"):
        self.h5_file_path = h5_file_path
        self.h5_file = None
        self.mode = mode
        self.dataset = dataset
        if keys is not None:
            self.keys = keys
        elif dataset == "LLVIP":
            with h5py.File(h5_file_path, 'r') as f:
                self.keys = sorted(list(f['ir'].keys()), key=lambda x: int(x))
        else:
            with h5py.File(h5_file_path, 'r') as f:
                self.keys = list(f['ir'].keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_file_path, 'r', swmr=True)
        sample_name = self.keys[idx]
        ir_bytes = self.h5_file['ir'][sample_name][()]
        vi_bytes = self.h5_file['vi'][sample_name][()]
        mask_bytes = self.h5_file['mask'][sample_name][()]
        textA = torch.from_numpy(self.h5_file['text_ir'][sample_name][()])
        textB = torch.from_numpy(self.h5_file['text_vi'][sample_name][()])

        ir_gray = cv2.imdecode(ir_bytes, cv2.IMREAD_GRAYSCALE)
        mask_gray = cv2.imdecode(mask_bytes, cv2.IMREAD_GRAYSCALE)
        IR = torch.from_numpy(ir_gray).float().unsqueeze(0) / 255.0
        Mask = torch.from_numpy(mask_gray).float().unsqueeze(0) / 255.0

        if self.mode == 'rgb':
            vi_bgr = cv2.imdecode(vi_bytes, cv2.IMREAD_COLOR)
            vi_rgb = cv2.cvtColor(vi_bgr, cv2.COLOR_BGR2RGB)
            VI = torch.from_numpy(vi_rgb.transpose(2, 0, 1)).float() / 255.0
        elif self.mode == "gray":
            vi_gray = cv2.imdecode(vi_bytes, cv2.IMREAD_GRAYSCALE)
            VI = torch.from_numpy(vi_gray).float().unsqueeze(0) / 255.0

        # if self.dataset == 'LLVIP':
        #     IR = F.interpolate(IR.unsqueeze(0), size=[480, 640], mode='area').squeeze(0)
        #     VI = F.interpolate(VI.unsqueeze(0), size=[480, 640], mode='area').squeeze(0)
        #     Mask = F.interpolate(Mask.unsqueeze(0), size=[480, 640], mode='area').squeeze(0)

        return IR, VI, textA, textB, Mask, sample_name


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
