import torch
import torch.nn as nn
from transformers import CvtModel
from typing import Tuple, Any, Optional
from models.basemodel import BaseModel
from models.decoder import Decoder


class EncoderWrapperCvT(nn.Module):
    """
    Wraps the HuggingFace CvtModel to extract feature maps and token embeddings.

    Returns:

    """
    def __init__(self,projection : nn.Module, pretrained_model_name: str = 'microsoft/cvt-13'):
        super().__init__()
        self.cvt = CvtModel.from_pretrained(pretrained_model_name)
        self.proj = projection
        for p in self.cvt.parameters():
            p.requires_grad = False
        for param in self.proj.parameters():
            param.requires_grad = True

    def forward(self, input_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        outputs = self.cvt(input_image)
        feature_maps = outputs.last_hidden_state  # [B, C, h, w]
        B, C, H, W = feature_maps.size()
        feature_maps = feature_maps.view(B,C, H*W)
        features = feature_maps.permute(0,2,1) # (B, 196, 384)

        encoded_seq = self.proj(features)
        return encoded_seq


class CvT_IC(BaseModel):
    """
    Image Captioning model with a CvT encoder and Transformer decoder.
    Inherits BaseModel to provide greedy/beam search with attention heatmap extraction.
    """
    def __init__(self,
                 vocab_size: int,
                 hidden_size: int = 512,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 cvt_model_name: str = 'microsoft/cvt-13'):
        super().__init__()
        # Encoder: pretrained CvT
        projection = nn.Linear(384, hidden_size)
        self.encoder = EncoderWrapperCvT(projection = projection ,pretrained_model_name=cvt_model_name)
        # Decoder: your existing Transformer decoder
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
        encoded_seq = self.encoder(input_image)

        decoded_seq, _ = self.decoder(
            target_seq = target_seq,
            encoder_output=encoded_seq,
            target_padding_mask = padding_mask,
            encoder_padding_mask = None
        )

        return decoded_seq

