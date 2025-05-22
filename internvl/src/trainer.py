# internvl/src/trainer.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModel, get_linear_schedule_with_warmup,
    TrainingArguments, Trainer
)
import yaml
from pathlib import Path
from typing import Dict, Optional
import logging
from tqdm import tqdm
import wandb
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InternVLTrainer:
    """Full training manager for InternVL models."""
    
    def __init__(self, 
                 model_path: str,
                 config_path: str = "config/training_config.yaml"):
        """
        Initialize trainer.
        
        Args:
            model_path: Path to pretrained model
            config_path: Path to training configuration
        """
        self.model_path = model_path
        self.config = self.load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        
    def load_config(self, config_path: str) -> Dict:
        """Load training configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer for training."""
        logger.info("Setting up model and tokenizer...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=False
        )
        
        # Load model
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Enable training mode
        self.model.train()
        
        logger.info("Model and tokenizer setup completed")
    
    def train(self, 
             train_dataloader: DataLoader,
             val_dataloader: Optional[DataLoader] = None,
             output_dir: str = "./models/trained/",
             resume_from_checkpoint: Optional[str] = None):
        """
        Train the model.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            output_dir: Output directory for trained model
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        logger.info("Starting training...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            betas=self.config['training']['optimizer']['betas'],
            eps=self.config['training']['optimizer']['eps']
        )
        
        total_steps = len(train_dataloader) * self.config['training']['num_epochs']
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config['training']['warmup_steps'],
            num_training_steps=total_steps
        )
        
        # Resume from checkpoint if provided
        start_epoch = 0
        if resume_from_checkpoint:
            checkpoint = torch.load(resume_from_checkpoint)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            logger.info(f"Resumed training from epoch {start_epoch}")
        
        # Initialize wandb if available
        try:
            wandb.init(
                project="internvl-training",
                config=self.config['training']
            )
            use_wandb = True
        except:
            use_wandb = False
            logger.warning("Wandb not available, proceeding without logging")
        
        # Training loop
        global_step = 0
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(start_epoch, self.config['training']['num_epochs']):
            logger.info(f"Epoch {epoch + 1}/{self.config['training']['num_epochs']}")
            
            # Training phase
            self.model.train()
            epoch_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    images=batch['images'],
                    labels=batch['labels']
                )
                
                loss = outputs.loss / self.config['training']['gradient_accumulation_steps']
                
                # Backward pass
                loss.backward()
                epoch_loss += loss.item()
                
                # Update weights
                if (batch_idx + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                # Update progress bar
                current_loss = loss.item() * self.config['training']['gradient_accumulation_steps']
                progress_bar.set_postfix({'loss': current_loss})
                
                # Logging
                if global_step % self.config['training']['logging']['steps'] == 0:
                    logger.info(f"Step {global_step}, Loss: {current_loss:.4f}")
                    
                    if use_wandb:
                        wandb.log({
                            "training_loss": current_loss,
                            "learning_rate": scheduler.get_last_lr()[0],
                            "step": global_step
                        })
                
                # Save checkpoint
                if global_step % self.config['training']['logging']['save_steps'] == 0:
                    self.save_checkpoint(
                        output_path / f"checkpoint-{global_step}",
                        epoch, optimizer, scheduler
                    )
            
            # Validation phase
            if val_dataloader is not None:
                val_loss = self.evaluate(val_dataloader)
                logger.info(f"Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}")
                
                if use_wandb:
                    wandb.log({
                        "validation_loss": val_loss,
                        "epoch": epoch + 1
                    })
                
                # Early stopping check
                if self.config['training']['early_stopping']['enabled']:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model
                        self.save_model(output_path / "best_model")
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= self.config['training']['early_stopping']['patience']:
                        logger.info("Early stopping triggered")
                        break
        
        # Save final model
        final_model_path = output_path / "final_model"
        self.save_model(final_model_path)
        logger.info(f"Training completed. Model saved to {final_model_path}")
        
        if use_wandb:
            wandb.finish()
    
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model on validation data."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    images=batch['images'],
                    labels=batch['labels']
                )
                
                total_loss += outputs.loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / num_batches
    
    def save_model(self, output_path: Path):
        """Save trained model."""
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(str(output_path))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(str(output_path))
        
        # Save training config
        with open(output_path / "training_config.yaml", 'w') as f:
            yaml.dump(self.config, f)
        
        logger.info(f"Model saved to {output_path}")
    
    def save_checkpoint(self, 
                       output_path: Path,
                       epoch: int,
                       optimizer: torch.optim.Optimizer,
                       scheduler):
        """Save training checkpoint."""
        output_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': self.config
        }
        
        torch.save(checkpoint, output_path / "checkpoint.pt")
        logger.info(f"Checkpoint saved to {output_path}")

