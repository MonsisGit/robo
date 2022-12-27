import logging
import pathlib
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from torchmetrics import Accuracy

from model import MLP
from dataloaders import my_collate, PoisitonsDataset

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)





def train():
    model = MLP(50, 50, 10, 3)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    model.train()
    accuracy = Accuracy(task="multiclass", num_classes=10)

    dataset = PoisitonsDataset(data_path=pathlib.Path('data/imgs_512.h5'))
    train_loader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=8,
        shuffle=False,
        collate_fn=my_collate)

    accs, losses = [], []
    for batch in tqdm(train_loader):
        model_input = torch.from_numpy(batch[0].flatten())
        target = batch[1]
        outputs = model(model_input)
        preds = outputs.sigmoid() > 0.5
        ce_loss = loss(preds, target)
        acc = accuracy(preds, target)
        losses.append(ce_loss)
        accs.append(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    train()
