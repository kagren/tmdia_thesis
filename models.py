from timm import create_model
import torch
from lightly.models.modules.heads import SimSiamPredictionHead, SimSiamProjectionHead
import torch.nn as nn


class SimClr(torch.nn.Module):
    def __init__(self, 
                 model_type: str,
                 img_size: tuple,
                 hidden_dim = 64):

        super().__init__()

        self.backbone = create_model(model_type,
                                  pretrained = True,
                                  num_classes = 0,
                                  drop_rate = 0,
                                  in_chans = 1)

        self.backbone_dim = self.backbone(torch.randn(1, 1, img_size[1], img_size[0])).shape[-1]

        self.fc = nn.Sequential(
                    nn.Linear(self.backbone_dim, 4 * hidden_dim),
                    nn.ReLU(),
                    nn.Linear(4 * hidden_dim, hidden_dim),
                )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x


class SimSiam(nn.Module):
    def __init__(self,
                 model_type: str,
                 img_size: tuple,
                 proj_hidden_dim = 1024,
                 pred_hidden_dim = 1024 // 4,
                 out_dim = 1024):
        super().__init__()

        self.backbone = create_model(model_type,
                                  pretrained = True,
                                  num_classes = 0,
                                  drop_rate = 0.,
                                  in_chans = 1)

        self.backbone_dim = self.backbone(torch.randn(1, 1, img_size[1], img_size[0])).shape[-1]

        self.projection_head = SimSiamProjectionHead(self.backbone_dim, proj_hidden_dim, out_dim)
        self.prediction_head = SimSiamPredictionHead(out_dim, pred_hidden_dim, out_dim)

        self.projection_nb = nn.BatchNorm1d(out_dim)

        self.out_dim = out_dim

    def forward(self, x):
        # get representations
        f = self.backbone(x).flatten(start_dim=1)
        # get projections
        z = self.projection_head(f)
        z = self.projection_nb(z)
        # get predictions
        p = self.prediction_head(z)
        # stop gradient
        z = z.detach()
        return z, p, f
