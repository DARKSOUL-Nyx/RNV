import torch
import torch.nn as nn
from .cnn_backbone import CNNBackbone
from .transformer_head import TransformerHead
from .fusion_blocks import ConcatFusion, AdditiveFusion, AttentionFusion, Patchify

class HybridModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        # CNN Backbone
        self.cnn = CNNBackbone(config["cnn"])
        cnn_out_dim = self.cnn.out_dim

        # Patchify
        self.patchify = Patchify(patch_size=config.get("patch_size", 16))

        # Transformer Backbone
        self.transformer = TransformerHead(config["transformer"])
        trans_out_dim = self.transformer.out_dim

        # Fusion
        fusion_type = config.get("fusion", "concat")
        if fusion_type == "concat":
            self.fusion = ConcatFusion()
            fused_dim = cnn_out_dim + trans_out_dim
        elif fusion_type == "add":
            self.fusion = AdditiveFusion()
            fused_dim = cnn_out_dim
        elif fusion_type == "attention":
            self.fusion = AttentionFusion(dim=min(cnn_out_dim, trans_out_dim))
            fused_dim = min(cnn_out_dim, trans_out_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # Classifier
        self.classifier = nn.Linear(fused_dim, config.get("num_classes", 10))

    def forward(self, x):
        x_cnn = self.cnn(x)
        x_patched = self.patchify(x_cnn)
        x_trans = self.transformer(x_patched)

        if isinstance(self.fusion, AttentionFusion):
            x_cnn_pooled = F.adaptive_avg_pool2d(x_cnn, (1, 1)).flatten(1).unsqueeze(1)
            x_trans = x_trans.unsqueeze(1)
            fused = self.fusion(x_cnn_pooled, x_trans).squeeze(1)
        else:
            x_cnn_pooled = F.adaptive_avg_pool2d(x_cnn, (1, 1)).flatten(1)
            fused = self.fusion(x_cnn_pooled, x_trans)


        return self.classifier(fused)
    