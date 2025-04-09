import sys
import os
import json
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.ICtransformer import ICTransformer
from utils.trainer import Trainer

def main():
    # Load config
    with open("config/train_config1.json", "r") as f:
        config = json.load(f)

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    # Create checkpoint directory
    os.makedirs(train_cfg["checkpoint_path"], exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = ICTransformer(
        image_size=model_cfg["image_size"],
        channels_in=3,
        vocab_size=10000,  # placeholder, updated after loading data
        patch_size=model_cfg["patch_size"],
        hidden_size=model_cfg["hidden_size"],
        num_layers=(model_cfg["num_encoder_layers"], model_cfg["num_decoder_layers"]),
        num_heads=(model_cfg["num_encoder_heads"], model_cfg["num_decoder_heads"]),
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        data_root=data_cfg["data_root"],
        caption_file=data_cfg["caption_file"],
        vocab_freq_threshold=data_cfg["vocab_threshold"],
        batch_size=train_cfg["batch_size"],
        num_epochs=train_cfg["num_epochs"],
        learning_rate=train_cfg["learning_rate"],
        save_freq=train_cfg["save_freq"],
        checkpoint_path=train_cfg["checkpoint_path"],
        use_mixed_precision=train_cfg["mixed_precision"],
        early_stopping_patience=train_cfg["early_stopping"],
        validation_split=data_cfg["validation_split"],
        num_workers=data_cfg["num_workers"],
        device=device
    )

    # Load data
    train_loader, val_loader = trainer.load_data()

    # Update decoder embedding and output layer
    model.decoder.embedding = nn.Embedding(
        trainer.vocab_size, model_cfg["hidden_size"], padding_idx=trainer.pad_idx
    )
    model.decoder.fc_out = nn.Linear(model_cfg["hidden_size"], trainer.vocab_size)

    # Initialize model weights, optimizers, etc.
    trainer.initialize_model()

    # Train the model
    trainer.train(resume_from=train_cfg["resume"])


if __name__ == "__main__":
    main()
