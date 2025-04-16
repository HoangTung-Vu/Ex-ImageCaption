import sys
import os
import json
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.CvTIC import CvT_IC
from utils.trainer import Trainer

def main():
    # Load config directly from the ICTransformer2 config file
    config_path = "config/train_config5.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    # Create checkpoint directory
    os.makedirs(train_cfg["checkpoint_path"], exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize ICTransformer2 model
    model = CvT_IC(
        vocab_size=10000,  # placeholder, updated after loading data
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_decoder_layers"],
        num_heads=model_cfg["num_decoder_heads"],
        cvt_model_name='microsoft/cvt-21'
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

    # Initialize model weights, optimizers, etc.
    trainer.initialize_model()

    # Train the model
    trainer.train(resume_from=train_cfg["resume"])

    print("Training of CETD completed!")


if __name__ == "__main__":
    main()