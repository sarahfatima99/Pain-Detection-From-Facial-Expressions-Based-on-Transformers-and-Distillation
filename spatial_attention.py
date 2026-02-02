import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.proj = nn.Linear(embed_dim, 1) # a smaller NN that looks at one patch at a time. outputs one scaler value per patch
        self.last_attn = None 

    def forward(self, x):
        attn = self.proj(x).squeeze(-1)      # Each patch has one raw impotance score
        attn = torch.softmax(attn, dim=1)    # converts scores into probabilities
        attn = attn.unsqueeze(-1)   
        self.last_attn = attn.detach()         # normalize accross the patch
        return x * (1 + attn)  #atten = {0,1} 1+attn = {1,2}. Each patch token is now scaled. 

