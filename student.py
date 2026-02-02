import torch
import torch.nn as nn
from transformer import TransformerBlock
from spatial_attention import SpatialAttention
from patch_embedding import PatchEmbedding

from config.config import IMG_SIZE, PATCH_SIZE, EMBED_DIM, DEPTH, NUM_CLASSES, NUM_HEADS




class StudentDeiT(nn.Module):
    def __init__(self):
        super().__init__()

        self.patch_embed = PatchEmbedding(PATCH_SIZE, EMBED_DIM)
        num_patches = (IMG_SIZE // PATCH_SIZE) ** 2

        self.cls_token  = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
        self.pos_embed  = nn.Parameter(
            torch.zeros(1, num_patches + 2, EMBED_DIM)
        )

        self.spatial_attn = SpatialAttention(EMBED_DIM)

        self.blocks = nn.ModuleList([
            TransformerBlock(EMBED_DIM, NUM_HEADS)
            for _ in range(DEPTH)
        ])

        self.norm = nn.LayerNorm(EMBED_DIM)

        self.cls_head  = nn.Linear(EMBED_DIM, NUM_CLASSES)
        self.dist_head = nn.Linear(EMBED_DIM, NUM_CLASSES)

    def forward(self, x):
        B = x.size(0)

        # Patch embedding
        x = self.patch_embed(x)              # (B, 196, 384) = (batch_size, number of patches, Embedd_dimension ) Each token (patch, class, distillation) is represented by a 384-dimensional vector.
        
        # Add tokens
        cls  = self.cls_token.expand(B, -1, -1)
        dist = self.dist_token.expand(B, -1, -1)
        x = torch.cat([cls, dist, x], dim=1) # (B, 198, 384)

        # Positional encoding
        x = x + self.pos_embed[:, :x.size(1)] # Positional embeddings inject location information into tokens.
        
        # Spatial attention (ALL tokens)
        
        

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
                
        x = self.blocks[-1](x)
        x = self.norm(x)

        # Extract tokens AFTER transformer
        x = self.spatial_attn(x[:, 2:]) #spatial attention to image patches

        cls_token_final  = x[:, 0]
        dist_token_final = x[:, 1]


        cls_logits  = self.cls_head(cls_token_final)
        dist_logits = self.dist_head(dist_token_final)
     

        return cls_logits, dist_logits
