from models.decoder import * 
from models.encoders import *
from models.basemodel import BaseModel

class ICTransformer(BaseModel):
    def __init__(self, image_size : int, channels_in : int, vocab_size : int, patch_size : int = 16, 
                 hidden_size : int = 256, num_layers : Tuple[int, int] = (2,2), num_heads : Tuple[int, int] = (4,4)):
        super(ICTransformer, self).__init__()
        self.encoder = VisionEncoder(
            image_size=image_size,
            channels_in=channels_in,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_layers=num_layers[0],
            num_heads=num_heads[0]
        )
        self.vocab_size = vocab_size
        self.decoder = Decoder(
            vocab_size=self.vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers[1],
            num_heads=num_heads[1]
        )
    
    def forward(self, 
                input_image : torch.Tensor,
                target_seq : torch.Tensor,
                padding_mask : Optional[torch.Tensor] = None):
        """
        Forward pass of the ICTransformer.
        Args:
            input_image (torch.Tensor): Input image tensor of shape (B, C, H, W).
            target_seq (torch.Tensor): Target sequence tensor of shape (B, N).
            padding_mask (Optional[torch.Tensor]): Padding mask for the target sequence. Shape (B, N).
        Return :
            torch.Tensor: Output tensor of shape (B, N, vocab_size).
        """
        encoded_seq = self.encoder(input_image)
        decoded_seq, _ = self.decoder(
            target_seq=target_seq,
            encoder_output=encoded_seq,
            target_padding_mask=padding_mask,
            encoder_padding_mask=None
        )

        return decoded_seq


    