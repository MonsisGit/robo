import logging
import pathlib
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torchmetrics.classification import BinaryAccuracy

from model import MLP
from dataloaders import collate_pos, PoisitonsDataset

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def train():
    model = MLP(30, 30, 10, 3)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = StepLR(optimizer, step_size=200, gamma=0.1)
    model.train()
    accuracy = BinaryAccuracy()
    epochs = 1000

    dataset_train = PoisitonsDataset(data_path=pathlib.Path("data/pos_train.pkl"),
                                     only_xyc=True)
    train_loader = DataLoader(
        dataset_train,
        batch_size=256,
        num_workers=8,
        shuffle=True,
        collate_fn=collate_pos)

    dataset_test = PoisitonsDataset(data_path=pathlib.Path("data/pos_test.pkl"),
                                    only_xyc=True)
    test_loader = DataLoader(
        dataset_test,
        batch_size=256,
        num_workers=8,
        shuffle=False,
        collate_fn=collate_pos)

    for e in tqdm(range(epochs)):
        accs, losses = [], []

        if e < 10:
            scheduler.optimizer.param_groups[0]['lr'] = \
                (0.01 + (e / 10)) * scheduler.optimizer.defaults['lr']
        if e == 10:
            scheduler.optimizer.param_groups[0]['lr'] = scheduler.optimizer.defaults['lr']

        for batch in train_loader:
            model_input = batch[0]
            target = batch[1]

            outputs = model(model_input)
            preds = outputs.sigmoid()
            ce_loss = criterion(preds, target)
            thresh_preds = (preds > 0.5).int()
            acc = accuracy(thresh_preds, target)
            losses.append(ce_loss.detach().float())
            accs.append(acc.detach().float())

            optimizer.zero_grad()
            ce_loss.backward()
            optimizer.step()
        scheduler.step()

        if e % 20 == 0:
            accs = []
            for batch in test_loader:
                model_input = batch[0]
                target = batch[1]

                outputs = model(model_input)
                preds = outputs.sigmoid()
                thresh_preds = (preds > 0.5).int()
                acc = accuracy(thresh_preds, target)
                accs.append(acc.detach().float())

            logger.info(f'\n Training Loss: {np.mean(losses):.3f} Acc: {np.mean(accs):.3f} \n Validation Acc: {np.mean(accs):.3f}')

    torch.save(model.state_dict(), "data/models/MLP.pth")
    logger.info('Model saved to {}'.format('data/models/MLP.pth'))


if __name__ == '__main__':
    train()
