import torch
import numpy as np
import os
from tqdm import tqdm
import pathlib
import logging
import pickle

from torch.utils.data import DataLoader

from image_dataloader import ImgDataset, my_collate
from model import get_yolo_model


def inference():
    logger = logging.getLogger(__name__)
    logger.info("Starting inference")

    dataset = ImgDataset(data_path=pathlib.Path('data/imgs_512.h5'),
                         target_path=pathlib.Path('data/targets.h5'))
    eval_loader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        shuffle=True,
        collate_fn=my_collate)

    model = get_yolo_model()
    pos, tars, confs = [], [], []
    for batch in tqdm(eval_loader):
        outputs = model(batch[0])
        for i in range(len(outputs)):
            xywh = outputs.xywh[i][:, :4].cpu().numpy()
            confidences = outputs.xywh[i][:, 4].cpu().numpy()
            #outputs.show()

            if xywh.shape[0] != 0:
                nm_cups = np.sum(batch[1], dtype=int)
                pos.append((xywh / 512)[0:nm_cups, :])
                tars.append(batch[1])
                confs.append(confidences[0:nm_cups])

    with open("data/extracted_pos.pkl", "wb") as fp:  # Pickling
        pickle.dump([pos, confs, tars], fp)


if __name__ == '__main__':
    inference()
