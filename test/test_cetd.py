import unittest
import torch
from models.CETD import ConvEnTransDe  # Update to your actual import path

class TestConvEnTransDe(unittest.TestCase):
    def setUp(self):
        self.image_size = 224
        self.channels_in = 3
        self.batch_size = 2
        self.seq_len = 10
        self.vocab_size = 1000
        self.hidden_size = 512

        # Dummy input image and target sequence
        self.dummy_image = torch.randn(self.batch_size, self.channels_in, self.image_size, self.image_size)
        self.dummy_target_seq = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        self.padding_mask = torch.zeros(self.batch_size, self.seq_len).bool()  # No padding

        # Initialize model
        self.model = ConvEnTransDe(
            image_size=self.image_size,
            channels_in=self.channels_in,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size
        )
        print(self.model.encoder)

    def test_output_shape(self):
        output = self.model(self.dummy_image, self.dummy_target_seq, self.padding_mask)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.vocab_size))

if __name__ == '__main__':
    unittest.main()
