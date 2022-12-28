import logging
import pathlib
import numpy as np
import PIL
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torchmetrics.classification import BinaryAccuracy

from model import load_mlp, get_yolo_model
from dataloaders import ImgDataset, collate_img, PoisitonsDataset, collate_pos
from yolo_inference import post_process_yolo

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


class CupPredictor:
    def __init__(self, model_path: pathlib.Path,
                 mlp_dims: dict):
        self.mlp_model = load_mlp(model_path, **mlp_dims)
        self.yolo_model = get_yolo_model()
        self.img_dataset = ImgDataset(data_path=pathlib.Path('data/imgs_512.h5'),
                                      target_path=pathlib.Path('data/targets.h5'))
        self.pos_dataset = PoisitonsDataset(data_path=pathlib.Path("data/pos_train.pkl"),
                                            only_xyc=True)

    def predict(self, image):
        if image is None:
            with torch.no_grad():
                pos, tars, confs = self._get_yolo_preds()
                cup_pos = self.get_mlp_preds(pos, tars, confs)
        return cup_pos

    def get_mlp_preds(self, pos, tars, confs):
        self.pos_dataset.concat_transform(pos, confs, tars,
                                          overwrite_class=True)
        batch = self.pos_dataset.get_item(0)
        batch = collate_pos(batch)

        outputs = self.mlp_model(batch[0])
        return (outputs.sigmoid() > 0.5).int()

    def _get_yolo_preds(self):
        out_train = []

        batch = self.img_dataset.get_item(0)
        batch = collate_img(batch)
        outputs = self.yolo_model(batch[0])
        pos, tars, confs = post_process_yolo(outputs, batch, out_train)
        return pos, tars, confs


if __name__ == '__main__':
    cup_predictor = CupPredictor(model_path=pathlib.Path('data/models/MLP.pth'),
                                 mlp_dims={'input_dim': 30,
                                           'hidden_dim': 30,
                                           'output_dim': 10,
                                           'num_layers': 3})
    cup_predictor.predict(image=None)
