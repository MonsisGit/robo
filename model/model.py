import os

import torch
from torch import nn
from torch.nn import functional as F


# Model
def get_yolo_model(model_type: str = 'yolov5l') -> torch.nn.Module:
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    model = torch.hub.load('ultralytics/yolov5', model_type, pretrained=True)
    model.conf = 0.4  # NMS confidence threshold
    model.iou = 0.35  # NMS IoU threshold
    agnostic = False  # NMS class-agnostic
    multi_label = False  # NMS multiple labels per box
    model.classes = [41]  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    max_det = 1000  # maximum number of detections per image
    amp = False  # Automatic Mixed Precision (AMP) inference
    return model


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    # input dime = (x+y+w+h+c)*10 = 50
    # output = 10
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i < self.num_layers - 1:
                x = F.relu(layer(x))
                x = self.dropout(x)
            else:
                x = layer(x)
        return x


