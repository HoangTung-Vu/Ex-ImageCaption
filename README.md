# Image Captioning Transformer

This repository contains an implementation of a Transformer-based architecture for image captioning. The model takes an image as input and generates a textual description (caption) of the image content.

## Repository Structure

\`\`\`
.
├── checkpoints/            # Model checkpoints
├── config/                 # Configuration files
│   ├── train_config1.json
│   └── train_config2.json
├── data/                   # Dataset files (not tracked by git)
├── eval_results/           # Evaluation results
├── exIC/                   # External image captioning resources
├── experiments_notebook/   # Jupyter notebooks for experiments
├── explain/                # Attention maps visualizations
├── models/                 # Model implementations
│   ├── basemodel.py        # Base model with common inference methods
│   ├── decoder.py          # Transformer decoder implementation
│   ├── encoders.py         # Vision encoder implementation
│   ├── ICTransformer.py    # Main model implementation
│   ├── ICTransformer2.py   # Enhanced model with pretrained ViT
│   ├── modelname.txt       # Model name definitions
│   └── modules.py          # Common modules (attention, transformer blocks)
├── scripts/                # Training and evaluation scripts
├── test/                   # Test scripts
└── utils/                  # Utility functions
    ├── dataloader.py       # Data loading utilities
    ├── evaluator.py        # Evaluation metrics
    ├── explain.py          # Explanation utilities
    ├── load.py             # Model loading utilities
    └── trainer.py          # Training utilities
\`\`\`

## Model Architecture

The repository implements two variants of image captioning transformers:

### ICTransformer

A transformer-based model with:
- Custom Vision Encoder: Processes images by extracting patches and encoding them with transformer blocks
- Transformer Decoder: Generates captions token by token with cross-attention to image features
- Sinusoidal positional embeddings for sequence information

### ICTransformer2

An enhanced version that:
- Uses a pretrained Vision Transformer (ViT) from the `timm` library as the encoder
- Freezes the ViT parameters for transfer learning
- Projects the ViT features to match the decoder's hidden size
- Uses the same transformer decoder architecture as ICTransformer