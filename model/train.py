import logging
import pickle

import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torchmetrics import Accuracy

from model import MLP

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def load_data():
    with open("data/extracted_pos.pkl", "rb") as fp:  # Unpickling
        pos, confs, targets = pickle.load(fp)
        for i in range(len(pos)):
            tmp = np.concatenate((pos[i], confs[i].reshape((-1, 1))), axis=1)
            pos[i] = np.pad(tmp, ((0, 10 - int(np.min((tmp.shape[0], 10)))), (0, 0)), 'constant', constant_values=0)
        labels = list()
        for _t in targets[::32]:
            labels.extend(_t)
        labels = torch.tensor(labels)[:min(len(labels), len(pos)), :]
        return pos, labels


def train():
    data, targets = load_data()
    model = MLP(50, 50, 10, 3)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    model.train()
    accuracy = Accuracy(task="multiclass", num_classes=10)

    accs, losses = [], []
    for idx, batch in tqdm(enumerate(data)):
        model_input = torch.from_numpy(batch.flatten())
        outputs = model(model_input)
        preds = outputs.sigmoid() > 0.5
        ce_loss = loss(preds, targets[idx])
        acc = accuracy(preds, targets[idx])
        losses.append(ce_loss)
        accs.append(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    train()
