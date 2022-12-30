import logging
import pathlib
import numpy as np
from PIL import Image, ImageDraw
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
                 mlp_dims: dict,
                 fov: list, verbose: bool = False):
        self.set_seed()
        self.mlp_model = load_mlp(model_path, **mlp_dims)
        self.yolo_model = get_yolo_model()
        self.img_dataset = ImgDataset(data_path=pathlib.Path('data/imgs_512.h5'),
                                      target_path=pathlib.Path('data/targets.h5'))
        self.pos_dataset = PoisitonsDataset(data_path=pathlib.Path("data/pos_train.pkl"),
                                            only_xyc=True)
        self.image_size = (512, 512)
        self.fov = fov
        self.verbose = verbose
        self.image = None

    def set_seed(self, seed: int = 42):
        torch.manual_seed(seed)
        np.random.seed(seed)

    def load_image(self, image: pathlib.Path,
                   image_size: tuple = (640, 640)) -> Image:
        self.image_size = image_size
        self.image = Image.open(image).resize(image_size).convert('RGB')
        if self.verbose:
            img = ImageDraw.Draw(self.image)
            img.rectangle(self.fov, fill=None, outline="blue")

        return self.image

    def predict(self, image: pathlib.Path) -> list:
        image = self.load_image(image)
        with torch.no_grad():
            pos, tars, confs = self._get_yolo_preds(image)
            cup_pos = self._get_mlp_preds(pos, tars, confs)

        if self.verbose:
            img = ImageDraw.Draw(self.image)
            img.text((0, 0), str(np.round(cup_pos, 3)), align="center", fill ="red")
            self.image.show()
        return cup_pos

    def _get_mlp_preds(self, pos, tars, confs):
        pos = np.concatenate((pos[:, 0:2], confs.reshape((-1, 1))), axis=1)
        pos = np.pad(pos, ((0, 10 - int(np.min((pos.shape[0], 10)))), (0, 0)), 'constant', constant_values=0)

        outputs = self.mlp_model(torch.tensor(pos).flatten())
        return outputs.sigmoid().tolist()

    def _get_yolo_preds(self, image: Image):
        outputs = self.yolo_model(image)
        xywh, confidences = self.post_process_yolo(outputs)
        return xywh, None, confidences

    def post_process_yolo(self, outputs) -> tuple:
        xywh = outputs.xywh[0][:, :4].cpu().numpy() / self.image_size[0]
        confidences = outputs.xywh[0][:, 4].cpu().numpy()
        pos_inds = self.check_bbox(outputs)
        return xywh[list(pos_inds), :], confidences[list(pos_inds)]

    def check_bbox(self, outputs):
        detections_cropped = [o['im'] for o in outputs.crop()]
        aspect_ratios = [d.shape[0] / d.shape[1] for d in detections_cropped]
        midpoints = [(d.shape[0] // 2, d.shape[1] // 2) for d in detections_cropped]
        pos_inds = set(np.where(np.array(aspect_ratios) > 1)[0])
        pos_inds_from_fov = set([idx for idx, m in enumerate(midpoints) if
                                 (self.fov[0] < m[0] < self.fov[2]) and (self.fov[1] < m[1] < self.fov[2])])

        if self.verbose:
            img = ImageDraw.Draw(self.image)
            for idx, _xyxy in enumerate(outputs.xyxy[0]):
                bbox_colour = 'green' if idx in pos_inds else 'red'
                img.rectangle(_xyxy[0:4].int().tolist(), fill=None, outline=bbox_colour)

        return pos_inds.intersection(pos_inds_from_fov)


if __name__ == '__main__':
    cup_predictor = CupPredictor(model_path=pathlib.Path('data/models/MLP.pth'),
                                 mlp_dims={'input_dim': 30,
                                           'hidden_dim': 30,
                                           'output_dim': 10,
                                           'num_layers': 3},
                                 fov=[200, 200, 450, 450],
                                 verbose=True)
    cup_predictor.predict(pathlib.Path('data/src_images/test/1011110101.1.jpg'))
