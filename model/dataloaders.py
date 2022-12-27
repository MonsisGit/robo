import os
import numpy as np
import logging
import pathlib
import h5py
from PIL import Image
import pickle

from torch.utils.data import Dataset
import torch

logger = logging.getLogger(__name__)


class ImgDataset(Dataset):

    def __init__(self, data_path: pathlib.Path,
                 target_path: pathlib.Path):
        self.data_path = data_path
        self.target_path = target_path
        self.data, self.target = self.load_data()

    def load_data(self):
        data = h5py.File(self.data_path, 'r')['X']
        target = h5py.File(self.target_path, 'r')['targets']
        return data, target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        _d = Image.fromarray(np.uint8(self.data[index] * 255)).convert('RGB')
        _t = self.target[index, ...]
        return [_d, _t]


class PoisitonsDataset(Dataset):

    def __init__(self, data_path="data/extracted_pos.pkl"):
        self.data_path = data_path
        self.data, self.target, self.used_inds = self.load_data()

    def load_data(self):
        with open(self.data_path, "rb") as fp:  # Unpickling
            pos, confs, targets, used_inds = pickle.load(fp)
            for i in range(len(pos)):
                tmp = np.concatenate((pos[i], confs[i].reshape((-1, 1))), axis=1)
                pos[i] = np.pad(tmp, ((0, 10 - int(np.min((tmp.shape[0], 10)))), (0, 0)), 'constant', constant_values=0)

            return pos, torch.tensor(targets), used_inds

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        _d = self.data[index]
        _t = self.target[index, ...]
        # _i = self.used_inds[index]
        return [_d, _t]


def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]
