
import os
import sys
import json
import torch
import numpy as np
import argparse
from tqdm import tqdm
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def load_model(config: Dict[str, Any], checkpoint_path: str, device: torch.device, model_type : str = 'ict2') -> Tuple[torch.nn.Module, Any]:
    """
    Load model from checkpoint.
    
    Args:
        config: Configuration dictionary
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on
        
    Returns:
        Tuple of (model, vocabulary)
    """
    # Import model dynamically
    from models.ICtransformer import ICTransformer
    from models.ICtransformer2 import ICTransformer2
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get vocabulary from checkpoint
    vocab = checkpoint['vocab']
    
    if model_type == 'ict2' : 
        # Create model
        model = ICTransformer2(
            image_size=config['model']['image_size'],
            channels_in=3,
            vocab_size=len(vocab),
            vit_model=config['model'].get('vit_model', 'vit_base_patch16_224'),
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_decoder_layers'],
            num_heads=config['model']['num_decoder_heads']
        )
    elif model_type == 'ict' :
        # Create model
        model = ICTransformer(
            image_size=config['model']['image_size'],
            patch_size = config['model']['patch_size'],
            channels_in=3,
            vocab_size=len(vocab),
            hidden_size=config['model']['hidden_size'],
            num_layers=(config['model']['num_encoder_layers'], config['model']['num_decoder_layers']),
            num_heads= (config['model']['num_encoder_heads'], config['model']['num_decoder_heads']),
        )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Model was trained for {checkpoint['epoch']} epochs")
    
    return model, vocab
