import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # label: 1 for similar, 0 for dissimilar
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        
        # Loss for similar pairs
        loss_similar = (label) * torch.pow(euclidean_distance, 2)
        
        # Loss for dissimilar pairs
        loss_dissimilar = (1 - label) * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0), 2
        )
        
        loss_contrastive = torch.mean(loss_similar + loss_dissimilar)
        return loss_contrastive