import os
import numpy as np
import logging
import pathlib
import h5py
from PIL import Image

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ImgDataset(Dataset):

    def __init__(self, data_path: pathlib.Path,
                 target_path: pathlib.Path):
        self.data_path = data_path
        self.target_path = target_path
        self.data, self.target = self.load_data()

    def load_data(self):
        data = h5py.File(self.data_path,'r')['X']
        target = h5py.File(self.target_path,'r')['targets']
        return data, target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        _d = Image.fromarray(np.uint8(self.data[index]*255)).convert('RGB')
        _t = self.target[index, ...]
        return [_d, _t]


def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]
