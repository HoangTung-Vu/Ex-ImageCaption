import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import torch
from models.encoders import VisionEncoder
from models.decoder import Decoder

class TestVisionEncoderDecoder(unittest.TestCase):

    def setUp(self):
        # Define model parameters
        self.image_size = 224
        self.patch_size = 16
        self.hidden_size = 256
        self.vocab_size = 1000
        self.num_layers = 2
        self.num_heads = 4
        self.batch_size = 8
        self.seq_length = 20

        # Initialize models
        self.encoder = VisionEncoder(
            image_size=self.image_size,
            patch_size=self.patch_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads
        )
        self.decoder = Decoder(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads
        )

    def test_encoder_import_and_instantiation(self):
        """Test if VisionEncoder can be imported and instantiated."""
        self.assertIsInstance(self.encoder, VisionEncoder)

    def test_decoder_import_and_instantiation(self):
        """Test if Decoder can be imported and instantiated."""
        self.assertIsInstance(self.decoder, Decoder)

    def test_encoder_forward_pass(self):
        """Test the forward pass of the VisionEncoder."""
        # Create a random image tensor with shape (batch_size, channels, height, width)
        image_tensor = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        encoder_output = self.encoder(image_tensor)
        # Expected output shape: (batch_size, num_patches, hidden_size)
        expected_shape = (self.batch_size, (self.image_size // self.patch_size) ** 2, self.hidden_size)
        self.assertEqual(encoder_output.shape, expected_shape)

    def test_decoder_forward_pass(self):
        """Test the forward pass of the Decoder."""
        # Create a random target sequence tensor with shape (batch_size, seq_length)
        target_seq = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        # Create a random encoder output tensor with shape (batch_size, num_patches, hidden_size)
        encoder_output = torch.randn(self.batch_size, (self.image_size // self.patch_size) ** 2, self.hidden_size)
        decoder_output = self.decoder(target_seq, encoder_output)
        # Expected output shape: (batch_size, seq_length, vocab_size)
        expected_shape = (self.batch_size, self.seq_length, self.vocab_size)
        self.assertEqual(decoder_output.shape, expected_shape)

    def test_encoder_decoder_integration(self):
        """Test the integration of VisionEncoder and Decoder."""
        # Create a random image tensor
        image_tensor = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        # Encode the image
        encoder_output = self.encoder(image_tensor)
        print(encoder_output.shape)
        # Create a random target sequence tensor with shape (batch_size, seq_length)
        target_seq = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        print(target_seq.shape)
        # Decode using the encoder output
        decoder_output = self.decoder(target_seq, encoder_output)
        print(decoder_output.shape)
        # Expected output shape: (batch_size, seq_length, vocab_size)
        expected_shape = (self.batch_size, self.seq_length, self.vocab_size)
        self.assertEqual(decoder_output.shape, expected_shape)

if __name__ == '__main__':
    unittest.main()
