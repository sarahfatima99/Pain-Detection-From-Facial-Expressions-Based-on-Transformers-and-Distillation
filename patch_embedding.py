import torch.nn as nn
import numpy as np


class PatchEmbedding(nn.Module): # converts the image into a sequence of tokens
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, patch_size, patch_size)

    def forward(self, x):  # Returns the patch embeddings ready for the Transformer block. Each image is now a sequence of tokens.
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x