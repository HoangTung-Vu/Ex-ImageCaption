import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import torch
from torch import nn
from models.modules import AttentionBlock, TransformerBlock

class TestTransformerComponents(unittest.TestCase):

    def setUp(self):
        self.batch_size = 2
        self.seq_len = 8
        self.hidden_size = 256
        self.num_heads = 4

        self.query = torch.rand(self.batch_size, self.seq_len, self.hidden_size)
        self.key = torch.rand(self.batch_size, self.seq_len, self.hidden_size)
        self.value = torch.rand(self.batch_size, self.seq_len, self.hidden_size)
        self.encoder_output = torch.rand(self.batch_size, self.seq_len, self.hidden_size)

    def test_attention_block_output_shape(self):
        attn = AttentionBlock(hidden_size=self.hidden_size, num_heads=self.num_heads)
        out = attn(self.query, self.key, self.value)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.hidden_size))

    def test_attention_block_output_shape_with_weights(self):
        attn = AttentionBlock(hidden_size=self.hidden_size, num_heads=self.num_heads, return_weights=True)
        out, weights = attn(self.query, self.key, self.value)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.hidden_size))
        self.assertEqual(weights.shape, (self.batch_size, self.seq_len, self.seq_len))

    def test_transformer_block_encoder_shape(self):
        block = TransformerBlock(hidden_size=self.hidden_size, num_heads=self.num_heads, is_decoder=False)
        out = block(self.query)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.hidden_size))

    def test_transformer_block_decoder_shape(self):
        block = TransformerBlock(hidden_size=self.hidden_size, num_heads=self.num_heads, is_decoder=True)
        out, weights = block(self.query, encoder_output=self.encoder_output)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.hidden_size))
        self.assertEqual(weights.shape, (self.batch_size, self.seq_len, self.seq_len))

if __name__ == "__main__":
    unittest.main()
