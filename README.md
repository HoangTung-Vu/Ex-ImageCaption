# Image Captioning Transformer

This repository contains an implementation of a Transformer-based architecture for image captioning. The model takes an image as input and generates a textual description (caption) of the image content.

## Repository Structure

```
.
├── checkpoints/            # Model checkpoints
├── config/                 # Configuration files
│   ├── train_config1.json  # Configuration for training 1
│   └── train_config2.json  # Configuration for training 2
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
```

## Model Architecture

The repository implements two variants of image captioning transformers:

### ICTransformer

A transformer-based model with:
- **Custom Vision Encoder**: Processes images by extracting patches and encoding them with transformer blocks.
- **Transformer Decoder**: Generates captions token by token with cross-attention to image features.
- **Sinusoidal Positional Embeddings**: Provides sequence information to the transformer.

### ICTransformer2

An enhanced version that:
- **Pretrained Vision Transformer (ViT) Encoder**: Uses a pretrained ViT from the `timm` library to extract image features.
  - The ViT encoder's parameters are frozen for transfer learning.
  - The extracted features are projected to match the decoder's hidden size.
- Uses the same **Transformer Decoder** architecture as ICTransformer.
