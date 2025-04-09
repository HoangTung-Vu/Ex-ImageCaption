import torch
import torch.nn as nn
from typing import Tuple, Optional
import timm

from models.decoder import Decoder
from models.basemodel import BaseModel 

class EncoderWrapper(nn.Module):
    def __init__(self, vit_model: timm.models.VisionTransformer, projection: nn.Module):
        super().__init__()
        self.vit = vit_model
        self.projection = projection
        for param in self.vit.parameters():
            param.requires_grad = False

        for param in self.projection.parameters():
            param.requires_grad = True 

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        """
        Extracts features using ViT, removes CLS token, and projects them.
        Ensures output is 3D: (B, SeqLen, Dim).
        """
        features = self.vit.forward_features(input_image)  # [B, 197, D] if CLS included

        if features.dim() == 2:
            features = features.unsqueeze(1)  # -> [B, 1, D]
        elif features.dim() != 3:
            raise ValueError(f"Unexpected feature dimension from ViT: {features.dim()}. Expected 2 or 3.")

        # Remove CLS token if present (common in ViT models)
        if features.shape[1] == 197:
            features = features[:, 1:, :]  # Drop CLS -> [B, 196, D]

        encoded_seq = self.projection(features)  # [B, SeqLen, hidden_size]
        return encoded_seq



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

        vit = timm.create_model(
            vit_model,
            pretrained=True,
            img_size=image_size,
            in_chans=channels_in
        )

        encoder_dim = vit.num_features
        projection = nn.Linear(encoder_dim, hidden_size) if encoder_dim != hidden_size else nn.Identity()
        self.encoder = EncoderWrapper(vit, projection)

        self.vocab_size = vocab_size
        self.decoder = Decoder(
            vocab_size=self.vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads
        )

        # Ensure decoder parameters require gradients (they should by default)
        for param in self.decoder.parameters():
            param.requires_grad = True

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
        encoded_seq = self.encoder(input_image) # Shape: [B, SeqLen, hidden_size]

        decoded_seq, _ = self.decoder(
            target_seq=target_seq,
            encoder_output=encoded_seq,
            target_padding_mask=padding_mask,
            encoder_padding_mask=None # Assuming encoder output is not padded
        )

        return decoded_seq
