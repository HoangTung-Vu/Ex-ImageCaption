import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from models.ICtransformer import ICTransformer
from utils.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Train Image Captioning Transformer")
    
    # Data parameters
    parser.add_argument("--data_root", type=str, default="data/flickr8k/Images", 
                        help="Path to image directory")
    parser.add_argument("--caption_file", type=str, default="data/flickr8k/captions.txt", 
                        help="Path to caption file")
    parser.add_argument("--vocab_threshold", type=int, default=5, 
                        help="Minimum word frequency for vocabulary")
    
    # Model parameters
    parser.add_argument("--image_size", type=int, default=224, 
                        help="Input image size")
    parser.add_argument("--patch_size", type=int, default=16, 
                        help="Vision transformer patch size")
    parser.add_argument("--hidden_size", type=int, default=512, 
                        help="Model hidden dimension size")
    parser.add_argument("--num_encoder_layers", type=int, default=3, 
                        help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=3, 
                        help="Number of decoder layers")
    parser.add_argument("--num_heads", type=int, default=8, 
                        help="Number of attention heads")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=30, 
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4, 
                        help="Learning rate")
    parser.add_argument("--save_freq", type=int, default=1, 
                        help="Checkpoint saving frequency (epochs)")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/", 
                        help="Path to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, 
                        help="Path to checkpoint to resume from")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="Number of data loading workers")
    parser.add_argument("--mixed_precision", action="store_true", 
                        help="Use mixed precision training")
    parser.add_argument("--early_stopping", type=int, default=5, 
                        help="Early stopping patience (epochs)")
    parser.add_argument("--validation_split", type=float, default=0.1, 
                        help="Fraction of data to use for validation")
    
    args = parser.parse_args()
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_path, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    print("Initializing model...")
    model = ICTransformer(
        image_size=args.image_size,
        channels_in=3,  # RGB images
        vocab_size=10000,  # This will be updated after loading data
        patch_size=args.patch_size,
        hidden_size=args.hidden_size,
        num_layers=(args.num_encoder_layers, args.num_decoder_layers),
        num_heads=args.num_heads
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        data_root=args.data_root,
        caption_file=args.caption_file,
        vocab_freq_threshold=args.vocab_threshold,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        save_freq=args.save_freq,
        checkpoint_path=args.checkpoint_path,
        use_mixed_precision=args.mixed_precision,
        early_stopping_patience=args.early_stopping,
        validation_split=args.validation_split,
        num_workers=args.num_workers,
        device=device
    )
    
    # Load data
    print("Loading data...")
    train_loader, val_loader = trainer.load_data()
    
    # Update model's vocabulary size
    model.decoder.embedding = nn.Embedding(
        trainer.vocab_size, 
        args.hidden_size, 
        padding_idx=trainer.pad_idx
    )
    model.decoder.fc_out = nn.Linear(args.hidden_size, trainer.vocab_size)
    
    # Initialize model components
    trainer.initialize_model()
    
    # Start training
    print("Starting training...")
    trainer.train(resume_from=args.resume)
    
    print("Training completed!")

if __name__ == "__main__":
    main()

