import torch
from torch.utils.data import Dataset
import os
from scipy.io import loadmat
import numpy as np


class MyData(Dataset):
    def __init__(self, data_dir, sort, length):
        super(MyData, self).__init__()

        if length == 2000:
            radar_path = os.path.join(data_dir, sort + 'Radar2000corr.npy')
            label_path = os.path.join(data_dir, sort + 'Label2000corr.npy')
            index_path = os.path.join(data_dir, sort + 'Index2000corr.npy')
            ecg_path = os.path.join(data_dir, sort + 'ECG2000corr.npy')

        else:
            radar_path = os.path.join(data_dir, sort + 'Radar640corr.npy')
            label_path = os.path.join(data_dir, sort + 'Label640corr.npy')
            index_path = os.path.join(data_dir, sort + 'Index640corr.npy')
            ecg_path = os.path.join(data_dir, sort + 'ECG640corr.npy')
            info_path = os.path.join(data_dir, sort + 'Info640corr.npy')

        self.radar = np.load(radar_path)
        self.label = np.load(label_path)
        self.index = np.load(index_path)
        self.ECG = np.load(ecg_path)
        self.info = np.load(info_path)

    def __getitem__(self, idx):

        x1 = self.index[idx][0]
        y1 = self.index[idx][1]
        x2 = self.index[idx][2]
        y2 = self.index[idx][3]
        x3 = self.index[idx][2]
        y3 = self.index[idx][3]

        # info
        x4 = self.index[idx][4]
        y4 = self.index[idx][5]

        radar_data = self.radar[:, x1: y1]
        label_data = self.label[:, x2: y2]
        ECG_data = self.ECG[:, x3: y3]
        info_data = self.info[x4]
        slice_num = int(y4)

        radar_data = torch.tensor(radar_data, dtype=torch.float)
        label_data = torch.tensor(label_data, dtype=torch.float)
        ECG_data = torch.tensor(ECG_data, dtype=torch.float)

        # ------------------information -----------------------------
        info_dice = {}
        info_dice['num'] = info_data[0]
        info_dice['diagnosis'] = info_data[1]
        info_dice['checkresult'] = info_data[2]
        info_dice['slice_num'] = slice_num

        # ------------------information end -------------------------

        sample = {'radar': radar_data, 'label': label_data, 'ECG': ECG_data, 'info': info_dice}
        return sample

    def __len__(self):
        return self.index.shape[0]

    def get_point_nums(self):
        return self.radar.shape[0]
