import torch
import torch.nn as nn
import torch.nn.functional as F
from .hybrid_model import HybridModel

class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = HybridModel(config)
        self.metric_head = config.get("metric_head", "cosine")

    def forward(self, img1, img2):
        f1 = self.encoder(img1)
        f2 = self.encoder(img2)

        if self.metric_head == "cosine":
            return F.cosine_similarity(f1, f2)
        elif self.metric_head == "l1":
            return torch.abs(f1 - f2)
        else:
            return torch.cat([f1, f2], dim=1)
