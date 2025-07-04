import os

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as transforms
import numpy as np 
from tqdm import tqdm
from typing import Tuple, Optional, List, Any, Callable, Type

from utils.dataloader import get_loader, FlickrDataset
from torch.utils.data import DataLoader

class Trainer : 
    def __init__(
            self,
            model : nn.Module,
            data_root : str,
            caption_file : str,
            vocab_freq_threshold : int = 1, 
            batch_size : int = 32,
            num_epochs : int = 10,
            learning_rate : float = 3e-4,
            save_freq : int = 2,
            checkpoint_path : str = "checkpoints/",
            use_mixed_precision : bool = False,
            early_stopping_patience : int = 3,
            validation_split : float = 0.2,
            num_workers : int = 4,
            device : Optional[torch.device] = None,
    ) :
        """
        Trainer class for training the ICTransformer model.
        Args:
            model (nn.Module): The model to be trained.
            data_root (str): Path to the root directory of the dataset.
            caption_file (str): Path to the caption file.
            vocab_freq_threshold (int): Minimum frequency for vocabulary words.
            batch_size (int): Batch size for training.
            num_epochs (int): Number of epochs to train.
            learning_rate (float): Learning rate for the optimizer.
            save_freq (int): Frequency of saving checkpoints.
            checkpoint_path (str): Path to save checkpoints.
            use_mixed_precision (bool): Whether to use mixed precision training.
            early_stopping_patience (int): Patience for early stopping.
            validation_split (float): Fraction of data to use for validation.
            num_workers (int): Number of workers for data loading.
        """
        self.model = model
        self.data_root = data_root
        self.caption_file = caption_file
        self.vocab_freq_threshold = vocab_freq_threshold
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.save_freq = save_freq
        self.checkpoint_path = checkpoint_path
        self.use_mixed_precision = use_mixed_precision
        self.early_stopping_patience = early_stopping_patience
        self.validation_split = validation_split
        self.num_workers = num_workers


        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        os.makedirs(self.checkpoint_path, exist_ok=True)

        # Initialize components to None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.dataset: Optional[FlickrDataset] = None
        self.vocab_size: Optional[int] = None
        self.pad_idx: Optional[int] = None
        self.criterion: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self.writer: Optional[SummaryWriter] = None

        self.scaler = torch.amp.GradScaler() if self.use_mixed_precision and torch.cuda.is_available() else None
        if self.use_mixed_precision and torch.cuda.is_available():
            print("Mixed precision training enabled.")
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5), # Data augmentation
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), # More augmentation
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    
    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        Load the dataset and create data loaders for training and validation.
        Returns:
            Tuple[DataLoader, DataLoader]: Training and validation data loaders.
        """

        print("Chuan bi du lieu...")

        full_loader, self.dataset = get_loader(
            root_folder=self.data_root,
            annotation_file=self.caption_file,
            freq_threshold=self.vocab_freq_threshold,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            transform=self.transform,
            img_cache_size=100
        )

        self.vocab_size = len(self.dataset.vocab)
        self.pad_idx = self.dataset.vocab.stoi["<PAD>"]
        print(f"Vocabulary size: {self.vocab_size}")    

        datasetsize = len(self.dataset)
        val_size = int(datasetsize * self.validation_split)
        train_size = datasetsize - val_size
        print(f"Dataset size: {datasetsize}, Train size: {train_size}, Validation size: {val_size}")

        train_dataset, val_dataset = torch.utils.data.random_split(self.dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

        collate_fn = full_loader.collate_fn

        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )
        print("Data loading and splitting complete.")
        return self.train_loader, self.val_loader

    def initialize_model(self) -> None:
        """
        Initialize the model, loss function, optimizer, and learning rate scheduler.
        """
        # Check if the model is ICTransformer or ICTransformer2
        if hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'embedding'):
            # Update the embedding layer with the correct vocabulary size
            self.model.decoder.embedding = nn.Embedding(
                self.vocab_size, 
                self.model.decoder.hidden_size, 
                padding_idx=self.pad_idx
            )
            
            # Update the output layer if it exists
            if hasattr(self.model.decoder, 'fc_out'):
                self.model.decoder.fc_out = nn.Linear(self.model.decoder.hidden_size, self.vocab_size)
        
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        

        params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"Training {len(params)} parameter groups (decoder and projection only)")

            
        self.optimizer = optim.Adam(params, lr=self.learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=1, verbose=True)
        self.writer = SummaryWriter(log_dir=os.path.join(self.checkpoint_path, "logs"))
        print("Model components initialized:")
        print(f"  Model: {self.model.__class__.__name__}")
        print(f"  Criterion: {self.criterion.__class__.__name__}")
        print(f"  Optimizer: {self.optimizer.__class__.__name__}")
        if self.scheduler:
             print(f"  Scheduler: {self.scheduler.__class__.__name__}")


    def save_checkpoint(self, epoch: int, val_loss : float, best_val_loss : float, is_best : bool) :
        """
        Save the model checkpoint.
        Args:
            epoch (int): Current epoch number.
            val_loss (float): Validation loss.
            best_val_loss (float): Best validation loss so far.
            is_best (bool): Whether this is the best model so far.
        """

        if self.model is None or self.optimizer is None or self.dataset is None:
            raise ValueError("Model, optimizer, or dataset not initialized.")
        
        os.makedirs(self.checkpoint_path, exist_ok=True)
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'vocab' : self.dataset.vocab,
            'model' : self.model.__class__.__name__
        }

        if self.scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.scaler is not None:
            checkpoint_data['scaler_state_dict'] = self.scaler.state_dict()
        

        latest_checkpoint_path = os.path.join(self.checkpoint_path, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint_data, latest_checkpoint_path)

        if is_best:
            best_checkpoint_path = os.path.join(self.checkpoint_path, "best_model.pth")
            torch.save(checkpoint_data, best_checkpoint_path)
            print(f"Best model saved at epoch {epoch} with validation loss: {val_loss:.4f}")
    
    def _prepare_batch(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepare a batch of data for training or validation.
        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A batch of data.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Prepared input and target tensors.
        """
        images, captions = batch
        images = images.to(self.device)
        captions = captions.to(self.device)

        input_caption = captions[:, :-1]
        target_caption = captions[:, 1:]
        
        padding_mask = (input_caption == self.pad_idx)
        

        return images, input_caption, target_caption, padding_mask

    def train_epoch(self, epoch: int) -> float:
        """
        Train the model for one epoch.
        Args:
            epoch (int): Current epoch number.
            Returns:
        float: Average loss for the epoch.
        """
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Train]")
        
        for batch_idx, batch in enumerate(progress_bar):
            self.optimizer.zero_grad()
            
            images, input_caption, target_caption, padding_mask = self._prepare_batch(batch) # New
            
            if self.use_mixed_precision and self.scaler is not None:
                with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                    outputs = self.model(images, input_caption, padding_mask = padding_mask)
                    loss = self.criterion(
                        outputs.reshape(-1, outputs.shape[-1]), 
                        target_caption.reshape(-1)
                    )
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images, input_caption, padding_mask = padding_mask)    
                loss = self.criterion(
                    outputs.reshape(-1, outputs.shape[-1]), 
                    target_caption.reshape(-1)
                )
                
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
            if batch_idx % 100 == 0:
                self.writer.add_scalar('Loss/train_step', loss.item(), 
                                      epoch * len(self.train_loader) + batch_idx)
        
        avg_loss = total_loss / len(self.train_loader)
        self.writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        return avg_loss
    
    def validate(self, epoch: int) -> float:
        """
        Validate the model on the validation set.
        Args:
            epoch (int): Current epoch number.
        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        total_loss = 0
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Val]")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                images, input_caption, target_caption, padding_mask = self._prepare_batch(batch)
                
                outputs = self.model(images, input_caption, padding_mask = padding_mask)
                loss = self.criterion(
                    outputs.reshape(-1, outputs.shape[-1]), 
                    target_caption.reshape(-1)
                )
                
                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(self.val_loader)
        self.writer.add_scalar('Loss/validation', avg_loss, epoch)
        return avg_loss
    
    def generate_sample_captions(self, epoch: int, num_samples: int = 5):
        """
        Generate sample captions for a few images in the validation set.
        Args:
            epoch (int): Current epoch number.
            num_samples (int): Number of samples to generate captions for.
        """
        if self.val_loader is None or self.dataset is None:
            return
            
        self.model.eval()
        samples = []
        
        with torch.no_grad():
            for i, (image, caption) in enumerate(self.val_loader):
                if i >= num_samples:
                    break
                    
                image = image[0].unsqueeze(0).to(self.device)
                
                generated_caption, _ = self.model.caption_image_greedy(
                    image=image,
                    vocabulary=self.dataset.vocab,
                    max_length=50
                )
                
                true_caption = [self.dataset.vocab.itos[idx.item()] for idx in caption[0] 
                               if idx.item() not in (self.dataset.vocab.stoi["<PAD>"], 
                                                   self.dataset.vocab.stoi["<SOS>"], 
                                                   self.dataset.vocab.stoi["<EOS>"])]
                
                samples.append({
                    'generated': ' '.join(generated_caption),
                    'true': ' '.join(true_caption)
                })
                
        print(f"Sample captions at epoch {epoch}:")
        for i, sample in enumerate(samples):
            print(f"Sample {i + 1}:")
            print(f"Generated: {sample['generated']}")
            print(f"True: {sample['true']}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found at {checkpoint_path}")
            return 0
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    def train(self, resume_from: Optional[str] = None) -> None:
        """
        Train the model.
        Args:
            resume_from (Optional[str]): Path to a checkpoint to resume training from.
        """
        if self.train_loader is None or self.val_loader is None:
            self.load_data()
            
        if self.criterion is None or self.optimizer is None:
            self.initialize_model()
            
        start_epoch = 0
        if resume_from is not None:
            start_epoch = self.load_checkpoint(resume_from)
            
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(start_epoch, self.num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()
                
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if epoch % self.save_freq == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, best_val_loss, is_best)
                
            if epoch % 2 == 0:
                self.generate_sample_captions(epoch)
                
            if patience_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
                
        self.writer.close()
        print("Training complete!")
