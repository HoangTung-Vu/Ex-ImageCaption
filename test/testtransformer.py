import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import unittest
import torch
from models.ICtransformer import ICTransformer

class DummyVocab:
    def __init__(self):
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "a": 3, "cat": 4, "on": 5, "mat": 6}
        self.itos = {v: k for k, v in self.stoi.items()}
    def __len__(self):
        return len(self.stoi)

class TestICTransformer(unittest.TestCase):

    def setUp(self):
        self.vocab = DummyVocab()
        self.model = ICTransformer(
            image_size=224,
            channels_in=3,
            vocab_size=len(self.vocab),
            patch_size=16,
            hidden_size=256,
            num_layers=(2, 2),
            num_heads=4
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.batch_size = 2
        self.seq_len = 10
        self.image = torch.randn(self.batch_size, 3, 224, 224).to(self.device)
        self.target_seq = torch.randint(0, len(self.vocab), (self.batch_size, self.seq_len)).to(self.device)

    def test_forward_output_shape(self):
        output = self.model(self.image, self.target_seq)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, len(self.vocab)))

    def test_greedy_caption_output(self):
        caption, attention = self.model.caption_image_greedy(self.image[0:1], self.vocab)
        self.assertIsInstance(caption, list)
        self.assertTrue(all(isinstance(word, str) for word in caption))
        self.assertEqual(len(caption), len(attention))

    def test_beam_search_output(self):
        caption = self.model.caption_image_beam_search(self.image[0:1], self.vocab, beam_size=3)
        self.assertIsInstance(caption, list)
        self.assertTrue(all(isinstance(word, str) for word in caption))
    
    def test_no_crash_on_short_seq(self):
        short_seq = torch.randint(0, len(self.vocab), (self.batch_size, 3)).to(self.device)
        output = self.model(self.image, short_seq)
        self.assertEqual(output.shape, (self.batch_size, 3, len(self.vocab)))

if __name__ == "__main__":
    unittest.main()
