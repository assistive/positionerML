# internvl/src/fine_tuner.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModel, get_linear_schedule_with_warmup,
    BitsAndBytesConfig, TrainingArguments, Trainer
)
from peft import (
    LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
)
import yaml
from pathlib import Path
from typing import Dict, Optional
import logging
from tqdm import tqdm
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InternVLFineTuner:
    """Fine-tuning manager for InternVL models using LoRA/QLoRA."""
    
    def __init__(self, 
                 model_path: str,
                 config_path: str = "config/training_config.yaml"):
        """
        Initialize fine-tuner.
        
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
        self.peft_model = None
        
    def load_config(self, config_path: str) -> Dict:
        """Load training configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer for fine-tuning."""
        logger.info("Setting up model and tokenizer...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=False
        )
        
        # Setup quantization if enabled
        quantization_config = None
        if self.config['training']['qlora']['enabled']:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config['training']['qlora']['quant_type'],
                bnb_4bit_use_double_quant=self.config['training']['qlora']['use_double_quant'],
                bnb_4bit_compute_dtype=torch.float16
            )
        
        # Load model
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Prepare model for training
        if quantization_config:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Setup LoRA
        if self.config['training']['lora']['enabled']:
            self.setup_lora()
        
    def setup_lora(self):
        """Setup LoRA configuration."""
        logger.info("Setting up LoRA...")
        
        lora_config = LoraConfig(
            r=self.config['training']['lora']['r'],
            lora_alpha=self.config['training']['lora']['alpha'],
            target_modules=self.config['training']['lora']['target_modules'],
            lora_dropout=self.config['training']['lora']['dropout'],
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()
        
        # Use PEFT model for training
        self.model = self.peft_model
    
    def fine_tune(self, 
                 train_dataloader: DataLoader,
                 val_dataloader: Optional[DataLoader] = None,
                 output_dir: str = "./models/fine_tuned/"):
        """
        Fine-tune the model.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            output_dir: Output directory for fine-tuned model
        """
        logger.info("Starting fine-tuning...")
        
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
        
        # Initialize wandb if available
        try:
            wandb.init(
                project="internvl-finetuning",
                config=self.config['training']
            )
            use_wandb = True
        except:
            use_wandb = False
            logger.warning("Wandb not available, proceeding without logging")
        
        # Training loop
        self.model.train()
        global_step = 0
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['training']['num_epochs']):
            logger.info(f"Epoch {epoch + 1}/{self.config['training']['num_epochs']}")
            
            epoch_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
            
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
                progress_bar.set_postfix({'loss': loss.item() * self.config['training']['gradient_accumulation_steps']})
                
                # Logging
                if global_step % self.config['training']['logging']['steps'] == 0:
                    current_loss = loss.item() * self.config['training']['gradient_accumulation_steps']
                    logger.info(f"Step {global_step}, Loss: {current_loss:.4f}")
                    
                    if use_wandb:
                        wandb.log({
                            "training_loss": current_loss,
                            "learning_rate": scheduler.get_last_lr()[0],
                            "step": global_step
                        })
                
                # Validation
                if (val_dataloader is not None and 
                    global_step % self.config['training']['logging']['eval_steps'] == 0):
                    val_loss = self.evaluate(val_dataloader)
                    logger.info(f"Validation Loss: {val_loss:.4f}")
                    
                    if use_wandb:
                        wandb.log({"validation_loss": val_loss, "step": global_step})
                    
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
                
                # Save checkpoint
                if global_step % self.config['training']['logging']['save_steps'] == 0:
                    self.save_model(output_path / f"checkpoint-{global_step}")
            
            # Early stopping check at epoch level
            if (self.config['training']['early_stopping']['enabled'] and 
                patience_counter >= self.config['training']['early_stopping']['patience']):
                break
        
        # Save final model
        final_model_path = output_path / "final_model"
        self.save_model(final_model_path)
        logger.info(f"Fine-tuning completed. Model saved to {final_model_path}")
        
        if use_wandb:
            wandb.finish()
    
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model on validation data."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
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
        """Save fine-tuned model."""
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save PEFT model if using LoRA
        if self.peft_model is not None:
            self.peft_model.save_pretrained(str(output_path))
        else:
            self.model.save_pretrained(str(output_path))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(str(output_path))
        
        # Save training config
        with open(output_path / "training_config.yaml", 'w') as f:
            yaml.dump(self.config, f)
        
        logger.info(f"Model saved to {output_path}")
    
    def load_fine_tuned_model(self, model_path: str):
        """Load a fine-tuned model."""
        logger.info(f"Loading fine-tuned model from {model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Load base model
        base_model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Load PEFT weights
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(base_model, model_path)
        
        logger.info("Fine-tuned model loaded successfully")

