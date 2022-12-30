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
        return self.get_item(index)

    def get_item(self, index):
        _d = Image.fromarray(np.uint8(self.data[index] * 255)).convert('RGB')
        _t = self.target[index, ...]
        return [_d, _t]


class PoisitonsDataset(Dataset):

    def __init__(self, data_path: pathlib.Path,
                 only_xyc: bool = False):
        self.data_path = data_path
        self.data, self.target = self.load_data()
        self.only_xyc = only_xyc

    def load_data(self):
        with open(self.data_path, "rb") as fp:  # Unpickling
            pos, confs, targets, used_inds = pickle.load(fp)

        return self.concat_transform(pos, confs, targets)

    def concat_transform(self, pos, confs, targets, overwrite_class_attributes=False):
        for i in range(len(pos)):
            tmp = np.concatenate((pos[i], confs[i].reshape((-1, 1))), axis=1)
            pos[i] = np.pad(tmp, ((0, 10 - int(np.min((tmp.shape[0], 10)))), (0, 0)), 'constant', constant_values=0)

        if overwrite_class_attributes:
            self.data = pos
            self.target = torch.tensor(targets)

        return pos, torch.tensor(targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.get_item(index)

    def get_item(self, index, with_target=True):
        _d = self.data[index]
        if self.only_xyc:
            _d = np.concatenate((_d[:, 0:2], _d[:, 4].reshape(-1, 1)), axis=1)

        _d + np.random.normal(0, 0.01, _d.shape[0] * _d.shape[1]).reshape(_d.shape)
        if with_target:
            _t = self.target[index, ...]
        else:
            _t = None
        return [_d, _t]


def collate_pos(batch):
    data = [item[0].flatten() for item in batch]
    target = [item[1] for item in batch]
    return [torch.tensor(data), torch.vstack(target).float()]


def collate_img(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]
