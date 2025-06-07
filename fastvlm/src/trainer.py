# fastvlm/src/trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    AutoTokenizer,
    AutoImageProcessor
)
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
from typing import Dict, Optional, List, Tuple, Union, Any
import logging
import wandb
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
import yaml
import time
from dataclasses import dataclass
import deepspeed
from fairscale.nn import checkpoint_wrapper
import torch.distributed as dist

from fastvlm_model import FastVLMModel, FastVLMConfig
from data_processor import DataProcessor, DataConfig, FastVLMDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Model
    model_name: str = "fastvlm-base"
    model_path: Optional[str] = None
    
    # Data
    train_data_path: str = None
    val_data_path: Optional[str] = None
    
    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Optimization
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    
    use_qlora: bool = False
    qlora_bits: int = 4
    
    use_gradient_checkpointing: bool = True
    mixed_precision: str = "fp16"  # fp16, bf16, or None
    
    # Training strategy
    distributed: bool = False
    deepspeed_config: Optional[str] = None
    fairscale_sharded: bool = False
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3
    
    output_dir: str = "./output"
    resume_from_checkpoint: Optional[str] = None
    
    # Wandb
    use_wandb: bool = True
    wandb_project: str = "fastvlm"
    wandb_run_name: Optional[str] = None
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


class FastVLMTrainer:
    """Trainer for FastVLM models with optimization strategies."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.accelerator = self._setup_accelerator()
        
        # Setup wandb
        if config.use_wandb and self.accelerator.is_main_process:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=config.__dict__
            )
        
        # Load model and tokenizers
        self.model, self.tokenizer, self.image_processor = self._load_model()
        
        # Setup data
        self.data_config = DataConfig()
        self.data_processor = DataProcessor(self.data_config)
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def _setup_accelerator(self) -> Accelerator:
        """Setup Accelerator for distributed training."""
        gradient_accumulation_steps = self.config.gradient_accumulation_steps
        
        if self.config.deepspeed_config:
            # Load DeepSpeed config
            with open(self.config.deepspeed_config, 'r') as f:
                deepspeed_config = json.load(f)
            
            accelerator = Accelerator(
                gradient_accumulation_steps=gradient_accumulation_steps,
                mixed_precision=self.config.mixed_precision,
                deepspeed_plugin=deepspeed_config
            )
        else:
            accelerator = Accelerator(
                gradient_accumulation_steps=gradient_accumulation_steps,
                mixed_precision=self.config.mixed_precision
            )
        
        return accelerator
    
    def _load_model(self) -> Tuple[FastVLMModel, AutoTokenizer, AutoImageProcessor]:
        """Load model with optimization strategies."""
        logger.info("Loading model and tokenizers...")
        
        # Load tokenizers
        if self.config.model_path:
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            image_processor = AutoImageProcessor.from_pretrained(self.config.model_path)
        else:
            # Use default tokenizers
            tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Load model
        if self.config.model_path:
            model = FastVLMModel.from_pretrained(self.config.model_path)
        else:
            config = FastVLMConfig()
            model = FastVLMModel(config)
        
        # Apply QLoRA if enabled
        if self.config.use_qlora:
            model = self._apply_qlora(model)
        
        # Apply LoRA if enabled
        elif self.config.use_lora:
            model = self._apply_lora(model)
        
        # Apply gradient checkpointing
        if self.config.use_gradient_checkpointing:
            model.gradient_checkpointing_enable()
            
            # Wrap layers with FairScale checkpoint
            if self.config.fairscale_sharded:
                for i, layer in enumerate(model.language_model.transformer.h):
                    model.language_model.transformer.h[i] = checkpoint_wrapper(layer)
        
        return model, tokenizer, image_processor
    
    def _apply_lora(self, model: FastVLMModel) -> FastVLMModel:
        """Apply LoRA to the model."""
        logger.info("Applying LoRA...")
        
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model
    
    def _apply_qlora(self, model: FastVLMModel) -> FastVLMModel:
        """Apply QLoRA with quantization."""
        logger.info(f"Applying {self.config.qlora_bits}-bit QLoRA...")
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # Apply LoRA
        model = self._apply_lora(model)
        
        return model
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        # Create data loaders
        train_loader, val_loader = self._create_dataloaders()
        
        # Setup optimizer and scheduler
        optimizer, scheduler = self._setup_optimization(len(train_loader))
        
        # Prepare for distributed training
        model, optimizer, train_loader, val_loader, scheduler = self.accelerator.prepare(
            self.model, optimizer, train_loader, val_loader, scheduler
        )
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training
            train_loss = self._train_epoch(model, train_loader, optimizer, scheduler)
            
            # Validation
            if val_loader is not None and (epoch + 1) % 1 == 0:
                val_loss = self._validate(model, val_loader)
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(model, optimizer, scheduler, epoch, "best")
                
                if self.config.use_wandb and self.accelerator.is_main_process:
                    wandb.log({
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "learning_rate": scheduler.get_last_lr()[0]
                    })
            
            # Save checkpoint
            if (epoch + 1) % 1 == 0:
                self._save_checkpoint(model, optimizer, scheduler, epoch, f"epoch_{epoch + 1}")
        
        logger.info("Training completed!")
        
        # Save final model
        self._save_final_model(model)
    
    def _create_dataloaders(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Create training and validation data loaders."""
        # Create datasets
        datasets = self.data_processor.create_datasets(
            train_path=self.config.train_data_path,
            val_path=self.config.val_data_path,
            tokenizer=self.tokenizer,
            image_processor=self.image_processor
        )
        
        # Create data loaders
        dataloaders = self.data_processor.create_dataloaders(
            datasets,
            batch_size=self.config.batch_size,
            distributed=self.config.distributed,
            num_replicas=self.accelerator.num_processes if self.config.distributed else None,
            rank=self.accelerator.process_index if self.config.distributed else None
        )
        
        train_loader = dataloaders['train']
        val_loader = dataloaders.get('val', None)
        
        return train_loader, val_loader
    
    def _setup_optimization(self, num_training_steps: int) -> Tuple[torch.optim.Optimizer, Any]:
        """Setup optimizer and learning rate scheduler."""
        # Get trainable parameters
        if hasattr(self.model, 'parameters'):
            params = self.model.parameters()
        else:
            params = filter(lambda p: p.requires_grad, self.model.parameters())
        
        # Create optimizer
        if self.config.use_qlora:
            # Use paged AdamW for QLoRA
            optimizer = bnb.optim.PagedAdamW8bit(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            optimizer = torch.optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        
        # Create scheduler
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        return optimizer, scheduler
    
    def _train_epoch(self, model, train_loader, optimizer, scheduler) -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(
            train_loader,
            desc="Training",
            disable=not self.accelerator.is_local_main_process
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            with self.accelerator.accumulate(model):
                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                
                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Update metrics
                total_loss += loss.detach().float()
                num_batches += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = total_loss / num_batches
                    progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
                    
                    if self.config.use_wandb and self.accelerator.is_main_process:
                        wandb.log({
                            "train_loss": avg_loss,
                            "learning_rate": scheduler.get_last_lr()[0],
                            "global_step": self.global_step
                        })
                
                self.global_step += 1
        
        return total_loss / num_batches
    
    def _validate(self, model, val_loader) -> float:
        """Validate the model."""
        model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", disable=not self.accelerator.is_local_main_process):
                outputs = model(**batch)
                loss = outputs.loss
                
                total_loss += loss.detach().float()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        logger.info(f"Validation Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def _save_checkpoint(self, model, optimizer, scheduler, epoch: int, tag: str):
        """Save training checkpoint."""
        if not self.accelerator.is_main_process:
            return
        
        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{tag}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        unwrapped_model = self.accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(checkpoint_dir)
        
        # Save tokenizers
        self.tokenizer.save_pretrained(checkpoint_dir)
        self.image_processor.save_pretrained(checkpoint_dir)
        
        # Save training state
        torch.save({
            'epoch': epoch,
            'global_step': self.global_step,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__
        }, checkpoint_dir / 'training_state.pt')
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def _save_final_model(self, model):
        """Save the final trained model."""
        if not self.accelerator.is_main_process:
            return
        
        final_dir = Path(self.config.output_dir) / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        unwrapped_model = self.accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(final_dir)
        
        # Save tokenizers
        self.tokenizer.save_pretrained(final_dir)
        self.image_processor.save_pretrained(final_dir)
        
        # Save config
        with open(final_dir / 'training_config.yaml', 'w') as f:
            yaml.dump(self.config.__dict__, f)
        
        logger.info(f"Final model saved to {final_dir}")
    
    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from a checkpoint."""
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        
        checkpoint_dir = Path(checkpoint_path)
        
        # Load model
        self.model = FastVLMModel.from_pretrained(checkpoint_dir)
        
        # Load training state
        state_path = checkpoint_dir / 'training_state.pt'
        if state_path.exists():
            state = torch.load(state_path, map_location='cpu')
            self.global_step = state['global_step']
            self.best_val_loss = state['best_val_loss']
            
            return state
        
        return None
