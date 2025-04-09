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

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.evaluator import Evaluator
from utils.dataloader import FlickrDataset, get_loader
import torchvision.transforms as transforms

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

def load_model(config: Dict[str, Any], checkpoint_path: str, device: torch.device) -> Tuple[torch.nn.Module, Any]:
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
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get vocabulary from checkpoint
    vocab = checkpoint['vocab']
    
    # Create model
    model = ICTransformer(
        image_size=config['model']['image_size'],
        channels_in=3,
        vocab_size=len(vocab),
        patch_size=config['model']['patch_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=(config['model']['num_encoder_layers'], config['model']['num_decoder_layers']),
        num_heads=(config['model']['num_encoder_heads'], config['model']['num_decoder_heads'])
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Model was trained for {checkpoint['epoch']} epochs")
    
    return model, vocab

def group_references_by_image(dataset: FlickrDataset) -> Tuple[List[str], List[List[str]]]:
    """
    Group multiple reference captions by image.
    
    Args:
        dataset: FlickrDataset instance
        
    Returns:
        Tuple of (unique_image_paths, grouped_references)
    """
    # Group captions by image
    image_to_captions = defaultdict(list)
    
    # Get all image paths and captions from the dataset
    for i in range(len(dataset.df)):
        img_id = dataset.imgs[i]
        caption = dataset.captions[i]
        image_to_captions[img_id].append(caption)
    
    # Convert to lists
    unique_images = list(image_to_captions.keys())
    grouped_references = [image_to_captions[img] for img in unique_images]
    
    # Get full image paths
    image_paths = [os.path.join(dataset.root, img) for img in unique_images]
    
    return image_paths, grouped_references

def run_evaluation(config_path: str, checkpoint_path: str, output_dir: str = "eval_results", num_samples: int = 10):
    """
    Run evaluation on the model.
    
    Args:
        config_path: Path to the configuration file
        checkpoint_path: Path to the checkpoint file
        output_dir: Directory to save evaluation results
        num_samples: Number of samples to visualize
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create evaluator
    evaluator = Evaluator(save_dir=output_dir)
    
    # Create tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))
    
    # Define image transforms
    transform = transforms.Compose([
        transforms.Resize((config['model']['image_size'], config['model']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Load dataset (without batching)
    print("Loading dataset...")
    dataset = FlickrDataset(
        root=config['data']['data_root'],
        captions_file=config['data']['caption_file'],
        transform=transform,
        freq_threshold=config['data']['vocab_threshold'],
        img_cache_size=100
    )
    
    # Group references by image
    print("Grouping references by image...")
    image_paths, grouped_references = group_references_by_image(dataset)
    print(f"Found {len(image_paths)} unique images with multiple references")
    
    # Load model
    print("Loading model...")
    model, vocab = load_model(config, checkpoint_path, device)
    
    # Generate captions for all images
    print("Generating captions...")
    hypotheses = []
    attention_maps = []
    
    with torch.no_grad():
        for i, img_path in enumerate(tqdm(image_paths)):
            # Load and preprocess image
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # Generate caption
            caption, attn_maps = model.caption_image_greedy(
                image=img_tensor,
                vocabulary=vocab,
                max_length=50
            )
            
            # Store results
            hypotheses.append(" ".join(caption))
            attention_maps.append(attn_maps)
    
    # Evaluate
    print("Calculating evaluation metrics...")
    results = evaluator.evaluate(grouped_references, hypotheses)
    
    # Save results
    evaluator.save_results(results)
    
    # Save sample captions for visualization
    sample_indices = np.random.choice(len(image_paths), min(num_samples, len(image_paths)), replace=False)
    evaluator.save_sample_captions(
        image_paths=image_paths,
        references=grouped_references,
        hypotheses=hypotheses,
        sample_indices=sample_indices,
        attention_maps=[attention_maps[i] for i in sample_indices]
    )
    
    # Visualize samples in tensorboard
    for i, idx in enumerate(sample_indices):
        # Load image
        img = Image.open(image_paths[idx])
        
        # Add image to tensorboard
        writer.add_image(f"Sample {i+1}/Image", transforms.ToTensor()(img), 0)
        
        # Add captions to tensorboard
        writer.add_text(f"Sample {i+1}/Generated", hypotheses[idx], 0)
        writer.add_text(f"Sample {i+1}/References", "\n".join(grouped_references[idx]), 0)
        
        # Add attention map if available
        if idx < len(attention_maps) and attention_maps[idx]:
            # Create heatmap visualization
            fig, ax = plt.subplots()
            ax.imshow(attention_maps[idx][0], cmap='hot', interpolation='nearest')
            ax.axis('off')
            writer.add_figure(f"Sample {i+1}/Attention", fig, 0)
    
    # Close tensorboard writer
    writer.close()
    print(f"Evaluation complete. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate image captioning model")
    parser.add_argument("--config", type=str, default="config/train_config1.json", help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, default = 'checkpoints/best_model.pth', help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="eval_results", help="Output directory")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to visualize")
    
    args = parser.parse_args()
    
    run_evaluation(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        num_samples=args.samples
    )
