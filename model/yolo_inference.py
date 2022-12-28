import torch
import numpy as np
import random
from tqdm import tqdm
import pathlib
import logging
import pickle

from torch.utils.data import DataLoader

from dataloaders import ImgDataset, collate_img
from model import get_yolo_model


def post_process_yolo(outputs, batch, out: list) -> list:
    pos, tars, confs = out
    for i in range(len(outputs)):
        xywh = outputs.xywh[i][:, :4].cpu().numpy()
        confidences = outputs.xywh[i][:, 4].cpu().numpy()
        # outputs.show()

        if xywh.shape[0] != 0:
            nm_cups = np.sum(batch[1][i], dtype=int)

            pos.append((xywh / 512)[0:nm_cups, :])
            confs.append(confidences[0:nm_cups])
            tars.append(batch[1][i])
            # used_inds.append(i)

    return [pos, tars, confs]


def save_pkl(data, path: pathlib.Path):
    with open(path, "wb") as fp:
        pickle.dump(data, fp)


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
        collate_fn=collate_img)

    model = get_yolo_model()
    out_train, out_test = [[], [], []], [[], [], []]

    for batch in tqdm(eval_loader):
        outputs = model(batch[0])

        if random.random() > 0.1:
            out_train = post_process_yolo(outputs, batch, out_train)
        else:
            out_test = post_process_yolo(outputs, batch, out_test)

    save_pkl(out_train, pathlib.Path("data/pos_train.pkl"))
    save_pkl(out_test, pathlib.Path("data/pos_test.pkl"))


if __name__ == '__main__':
    inference()
