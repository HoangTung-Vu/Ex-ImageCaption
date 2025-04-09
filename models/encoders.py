import torch
import torch.nn as nn 
from models.modules import * 

class VisionEncoder(nn.Module):
    def __init__(self, image_size : int = 224, channels_in : int = 3, patch_size : int = 16, hidden_size = 256, 
                 num_layers : int = 2, num_heads : int = 4):
        super(VisionEncoder, self).__init__()
        if image_size % patch_size != 0:
            raise ValueError("Image size must be divisible by patch size.")
        
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels_in * patch_size * patch_size

        self.patch_proj = nn.Linear(patch_dim, hidden_size)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, hidden_size)) # [1, num_patches, hidden_size]
        self.dropout = nn.Dropout(0.3)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size=hidden_size, num_heads=num_heads, drop_out=0.3, is_decoder=False)
            for _ in range(num_layers)
        ])

        self.norm_out = nn.LayerNorm(hidden_size)

    def forward(
            self, 
            image_tensor : torch.Tensor,
        ) -> torch.Tensor :

        batch_size = image_tensor.shape[0]
        patches = extract_patches(image_tensor, self.patch_size)
        patch_emb = self.patch_proj(patches)
        x = patch_emb

        x = x + self.pos_embedding
        x = self.dropout(x)

        for block in self.blocks : 
            x = block(x)

        x = self.norm_out(x)
        return x 
