import timm
import torch.nn as nn

class CNNBackbone(nn.Module):
    def __init__(self, name="resnet50", pretrained=True):
        super().__init__()
        self.model = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
        self.out_dim = self.model.num_features

    def forward(self, x):
        return self.model(x)   # (B, out_dim)
