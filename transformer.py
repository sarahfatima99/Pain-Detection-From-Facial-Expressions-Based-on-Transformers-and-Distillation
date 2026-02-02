import torch
import torch.nn as nn




class TransformerBlock(nn.Module): # dim → the dimensionality of the input feature vectors.
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim) # Normalizes the input across the feature dimension. Helps stabilize training and improve convergence.
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential( # This is a feed-forward network inside the Transformer block.
            nn.Linear(dim, dim * 4), # this expands dimension. This expansion lets the model capture more complex interactions between features.
            nn.GELU(), # applies non linear activation which helps to leanr non linearity of features
            nn.Linear(dim * 4, dim) # reduces back to original shape so that it can be added to residual connection
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), # inputs: query, key, value are all the same (x_norm) = self-attention. Multihead attention allows each token to look at other tokens and decide how much to “attend” to them.
                          self.norm1(x),
                          self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x)) # Normalize the token features again: self.norm2(x). Pass through MLP (feed-forward network) and expands each token features with non linear transformation. Now each token has learned complex non-linear relationships from its own features.
        return x
