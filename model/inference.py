import torch
import numpy as np
import random
from tqdm import tqdm
import pathlib
import logging
import pickle

from torch.utils.data import DataLoader

from dataloaders import ImgDataset, my_collate
from model import get_yolo_model


def inference():
    logger = logging.getLogger(__name__)
    logger.info("Starting inference")

    dataset = ImgDataset(data_path=pathlib.Path('data/imgs_512.h5'),
                         target_path=pathlib.Path('data/targets.h5'))
    eval_loader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=8,
        shuffle=False,
        collate_fn=my_collate)

    model = get_yolo_model()
    pos, tars, confs, used_inds = [], [], [], []
    pos_test, tars_test, confs_test, used_inds_test = [], [], [], []
    for batch in tqdm(eval_loader):
        outputs = model(batch[0])
        for i in range(len(outputs)):
            xywh = outputs.xywh[i][:, :4].cpu().numpy()
            confidences = outputs.xywh[i][:, 4].cpu().numpy()
            # outputs.show()

            if xywh.shape[0] != 0:
                nm_cups = np.sum(batch[1][i], dtype=int)

                if random.random() > 0.1:
                    pos.append((xywh / 512)[0:nm_cups, :])
                    confs.append(confidences[0:nm_cups])
                    tars.append(batch[1][i])
                    used_inds.append(i)
                else:
                    pos_test.append((xywh / 512)[0:nm_cups, :])
                    tars_test.append(batch[1][i])
                    confs_test.append(confidences[0:nm_cups])
                    used_inds_test.append(i)

    with open("data/pos_train.pkl", "wb") as fp:  # Pickling
        pickle.dump([pos, confs, tars, used_inds], fp)

    with open("data/pos_test.pkl", "wb") as fp:
        pickle.dump([pos_test, confs_test, tars_test, used_inds_test], fp)


if __name__ == '__main__':
    inference()
