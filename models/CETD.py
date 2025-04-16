import torch
import torch.nn as nn
import torchvision.models as model
from torchvision.models import ResNet50_Weights
from PIL import Image
from typing import Tuple, Optional

from models.decoder import Decoder
from models.basemodel import BaseModel 

def test(resnet, input_image):
    """
    Test function to check the shape of the output tensor.
    Args:
        resnet: Pre-trained ResNet model.
        projection: Projection layer.
        input_image: Input image tensor.
    """
    # Extract features using ResNet
    backbone = nn.Sequential(*list(resnet.children())[:-2])  # Up to layer4
    features = backbone(input_image)
    B, C, H, W = features.size()
    return C
    


class EncoderWrapper(nn.Module):
    def __init__(self, resnet : model.ResNet, projection : nn.Module):
        super().__init__()
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Up to layer4
        self.projection = projection
        for param in self.backbone.parameters():
            param.requires_grad = False 

        for param in self.projection.parameters():
            param.requires_grad = True
    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        """
        Extracts features using ResNet and projects them.
        Ensures output is 3D: (B, C, H*W).
        """
        features = self.backbone(input_image)
        B, C, H, W = features.size()
        features = features.view(B, C, H*W)
        features_permuted = features.permute(0, 2, 1)

        encoded_seq = self.projection(features_permuted)  # [B, H*W, C] -> [B, SeqLen, hidden_size]
        return encoded_seq


class ConvEnTransDe(BaseModel):
    def __init__(self, image_size : int, channels_in : int,  vocab_size: int, hidden_size: int = 512,
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
        super(ConvEnTransDe, self).__init__()
        dumb_img = torch.randn(1, channels_in, image_size, image_size)
        print(f"Dummy image shape: {dumb_img.shape}")

        # Load ResNet-50 with default pre-trained weights
        resnet = model.resnet50(weights=ResNet50_Weights.DEFAULT)
        dim_to_proj = test(resnet, dumb_img)
        projection = nn.Linear(dim_to_proj, hidden_size)
        self.encoder = EncoderWrapper(resnet, projection)

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
        