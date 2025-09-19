import torch
import torch.nn as nn
import torch.nn.functional as F

class ConcatFusion(nn.Module):
    def forward(self, x_cnn, x_trans):
        # Simple concatenation along feature dim
        return torch.cat([x_cnn, x_trans], dim=1)

class AdditiveFusion(nn.Module):
    def forward(self, x_cnn, x_trans):
        # Element-wise addition (requires same shape)
        return x_cnn + x_trans

class AttentionFusion(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x_cnn, x_trans):
        # Expect shape: (B, seq_len, dim)
        # CNN features need to be reshaped into seq form
        out, _ = self.attn(x_cnn, x_trans, x_trans)
        return out

class Patchify(nn.Module):
    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
    def forward(self, x):
        B, C, H, W = x.shape
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        return patches.contiguous().view(B, -1, C * self.patch_size * self.patch_size)
