import torch
import torch.nn as nn
from typing import Tuple, Optional
import timm

from models.decoder import Decoder
from models.basemodel import BaseModel

class ICTransformer2(BaseModel):
    def __init__(self, image_size: int, channels_in: int, vocab_size: int, 
                 vit_model: str = "vit_base_patch16_224", hidden_size: int = 512, 
                 num_layers: int = 2, num_heads: int = 8):
        """
        ICTransformer2 with pre-trained ViT encoder.
        
        Args:
            image_size: Input image size (assumed square)
            channels_in: Number of input channels
            vocab_size: Size of the vocabulary
            vit_model: Name of the pre-trained ViT model from timm
            hidden_size: Hidden size for the decoder
            num_layers: Number of layers in the decoder
            num_heads: Number of attention heads in the decoder
        """
        super(ICTransformer2, self).__init__()
        
        # Load pre-trained ViT model
        self.encoder = timm.create_model(
            vit_model, 
            pretrained=True, 
            img_size=image_size,
            in_chans=channels_in
        )
        
        # Get the feature dimension from the encoder
        encoder_dim = self.encoder.num_features
        
        # Create a projection layer if encoder dimension doesn't match hidden_size
        self.projection = nn.Linear(encoder_dim, hidden_size) if encoder_dim != hidden_size else nn.Identity()
        
        # Freeze the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.vocab_size = vocab_size
        self.decoder = Decoder(
            vocab_size=self.vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads
        )
    
    def forward(self, 
                input_image: torch.Tensor,
                target_seq: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None):
        """
        Forward pass of the ICTransformer2.
        
        Args:
            input_image: Input image tensor of shape (B, C, H, W)
            target_seq: Target sequence tensor of shape (B, N)
            padding_mask: Padding mask for the target sequence. Shape (B, N)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, N, vocab_size)
        """
        # Extract features using the pre-trained ViT
        # The output is typically [B, encoder_dim]
        features = self.encoder.forward_features(input_image)
        
        # Reshape if needed - ViT typically outputs [B, num_patches + 1, embed_dim]
        # where the +1 is for the class token
        if features.dim() == 2:
            # If the output is [B, encoder_dim], reshape to [B, 1, encoder_dim]
            features = features.unsqueeze(1)
        
        # Project features to match decoder hidden size
        encoded_seq = self.projection(features)
        
        # Decode
        decoded_seq, _ = self.decoder(
            target_seq=target_seq,
            encoder_output=encoded_seq,
            target_padding_mask=padding_mask,
            encoder_padding_mask=None
        )
        
        return decoded_seq
