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
      - feature_maps: [B, C, h, w]
      - tokens:       [B, num_patches, C]
    """
    def __init__(self, pretrained_model_name: str = 'microsoft/cvt-13'):
        super().__init__()
        self.cvt = CvtModel.from_pretrained(pretrained_model_name)
        for p in self.cvt.parameters():
            p.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pixel_values: [B, 3, H, W], preprocessed to match CvT input
        Returns:
            feature_maps: [B, C, h, w]
            tokens:       [B, h*w, C]
        """
        outputs = self.cvt(pixel_values=pixel_values)
        feature_maps = outputs.last_hidden_state  # [B, C, h, w]
        B, C, h, w = feature_maps.shape
        tokens = feature_maps.flatten(2).transpose(1, 2)  # [B, h*w, C]
        return feature_maps, tokens


class CvTConvEnTransDe(BaseModel):
    """
    Image Captioning model with a CvT encoder and Transformer decoder.
    Inherits BaseModel to provide greedy/beam search with attention heatmap extraction.
    """
    def __init__(self,
                 vocab_size: int,
                 hidden_size: int = 384,
                 num_layers: int = 2,
                 num_heads: int = 6,
                 cvt_model_name: str = 'microsoft/cvt-13'):
        super().__init__()
        # Encoder: pretrained CvT
        self.encoder = EncoderWrapperCvT(pretrained_model_name=cvt_model_name)
        # Decoder: your existing Transformer decoder
        self.decoder = Decoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads
        )

    def forward(self,
                input_image: torch.Tensor,
                target_seq: torch.Tensor,
                target_padding_mask: Optional[torch.Tensor] = None,
                encoder_padding_mask: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, Any]:
        # Encode image into conv feature maps and tokens
        feature_maps, encoder_tokens = self.encoder(input_image)
        # Decode with cross-attention; returns logits and attention weights
        logits, attn_weights = self.decoder(
            target_seq=target_seq,
            encoder_output=encoder_tokens,
            target_padding_mask=target_padding_mask,
            encoder_padding_mask=encoder_padding_mask
        )
        return logits, attn_weights
