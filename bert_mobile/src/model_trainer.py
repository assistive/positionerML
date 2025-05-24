# bert_mobile/src/bert_trainer.py

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple, Any
import logging
import yaml
from pathlib import Path
import numpy as np
from tqdm import tqdm
import time
import math

from transformers import (
    AutoTokenizer, AutoModel, AutoConfig, AutoModelForSequenceClassification,
    BertTokenizer, BertModel, BertConfig, BertForSequenceClassification,
    Trainer, TrainingArguments,
    DataCollatorWithPadding, DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BERTDataset(Dataset):
    """Custom dataset for BERT training."""
    
    def __init__(self, texts: List[str], labels: Optional[List[int]], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item

class BERTMobileTrainer:
    """Train and fine-tune BERT models optimized for mobile deployment."""
    
    def __init__(self, 
                 model_path: str,
                 config_path: str = "config/training_config.yaml",
                 custom_vocab_path: Optional[str] = None):
        """
        Initialize BERT trainer.
        
        Args:
            model_path: Path to base model
            config_path: Path to training configuration
            custom_vocab_path: Path to custom vocabulary file
        """
        self.model_path = model_path
        self.custom_vocab_path = custom_vocab_path
        self.config = self.load_config(config_path)
        
        # Initialize model components
        self.model = None
        self.tokenizer = None
        self.teacher_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training state
        self.training_state = {
            'epoch': 0,
            'step': 0,
            'best_loss': float('inf'),
            'best_accuracy': 0.0
        }
        
        # Metrics tracking
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rate': []
        }
    
    def load_config(self, config_path: str) -> Dict:
        """Load training configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_model_and_tokenizer(self, task_type: str = "classification", num_labels: int = 2):
        """Setup model and tokenizer for training."""
        logger.info("Setting up model and tokenizer...")
        
        # Load configuration
        model_config = AutoConfig.from_pretrained(self.model_path)
        
        # Load tokenizer
        if self.custom_vocab_path:
            logger.info(f"Using custom vocabulary: {self.custom_vocab_path}")
            self.tokenizer = BertTokenizer.from_pretrained(
                self.model_path,
                vocab_file=self.custom_vocab_path,
                do_lower_case=self.config['data'].get('do_lower_case', True)
            )
            # Update vocabulary size in config
            model_config.vocab_size = len(self.tokenizer.vocab)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Load model based on task type
        if task_type == "classification":
            model_config.num_labels = num_labels
            if self.custom_vocab_path:
                self.model = BertForSequenceClassification.from_pretrained(
                    self.model_path,
                    config=model_config
                )
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_path,
                    config=model_config
                )
        else:  # language modeling
            if self.custom_vocab_path:
                self.model = BertModel.from_pretrained(
                    self.model_path,
                    config=model_config
                )
            else:
                self.model = AutoModel.from_pretrained(
                    self.model_path,
                    config=model_config
                )
        
        # Resize token embeddings if using custom vocabulary
        if self.custom_vocab_path:
            self.model.resize_token_embeddings(len(self.tokenizer.vocab))
        
        # Apply mobile optimizations if configured
        if self.config['training'].get('mobile_training', {}).get('enabled', False):
            self.apply_mobile_optimizations()
        
        # Move to device
        self.model.to(self.device)
        
        logger.info(f"Model loaded on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        logger.info(f"Vocabulary size: {len(self.tokenizer.vocab):,}")
    
    def apply_mobile_optimizations(self):
        """Apply mobile-specific optimizations to the model."""
        logger.info("Applying mobile optimizations...")
        
        mobile_config = self.config['training']['mobile_training']
        
        # Enable knowledge distillation if specified
        if mobile_config.get('knowledge_distillation', False):
            self.setup_knowledge_distillation()
        
        # Reduce model size if specified
        if mobile_config.get('reduce_layers', False):
            self.reduce_model_layers()
        
        # Apply layer freezing
        freeze_layers = self.config['training']['fine_tuning'].get('freeze_encoder_layers', 0)
        if freeze_layers > 0:
            self.freeze_encoder_layers(freeze_layers)
        
        # Freeze embeddings if specified
        if self.config['training']['fine_tuning'].get('freeze_embeddings', False):
            self.freeze_embeddings()
    
    def setup_knowledge_distillation(self):
        """Setup knowledge distillation from a teacher model."""
        teacher_model_name = self.config['training']['mobile_training']['teacher_model']
        
        logger.info(f"Setting up knowledge distillation with teacher: {teacher_model_name}")
        
        # Load teacher model
        if hasattr(self.model, 'num_labels'):  # Classification model
            self.teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_model_name)
        else:
            self.teacher_model = AutoModel.from_pretrained(teacher_model_name)
            
        self.teacher_model.to(self.device)
        self.teacher_model.eval()
        
        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def reduce_model_layers(self):
        """Reduce the number of layers in the model for mobile optimization."""
        mobile_layers = self.config['training']['mobile_training'].get('num_layers', 6)
        
        if hasattr(self.model, 'bert') and hasattr(self.model.bert, 'encoder'):
            encoder = self.model.bert.encoder
        elif hasattr(self.model, 'encoder'):
            encoder = self.model.encoder
        else:
            logger.warning("Could not find encoder to reduce layers")
            return
        
        if hasattr(encoder, 'layer'):
            original_layers = len(encoder.layer)
            if mobile_layers < original_layers:
                logger.info(f"Reducing layers from {original_layers} to {mobile_layers}")
                encoder.layer = encoder.layer[:mobile_layers]
                
                # Update config
                if hasattr(self.model.config, 'num_hidden_layers'):
                    self.model.config.num_hidden_layers = mobile_layers
    
    def freeze_encoder_layers(self, num_layers: int):
        """Freeze the first N encoder layers."""
        logger.info(f"Freezing first {num_layers} encoder layers")
        
        if hasattr(self.model, 'bert') and hasattr(self.model.bert, 'encoder'):
            encoder = self.model.bert.encoder
        elif hasattr(self.model, 'encoder'):
            encoder = self.model.encoder
        else:
            logger.warning("Could not find encoder to freeze layers")
            return
        
        if hasattr(encoder, 'layer'):
            for i in range(min(num_layers, len(encoder.layer))):
                for param in encoder.layer[i].parameters():
                    param.requires_grad = False
                logger.info(f"Froze layer {i}")
    
    def freeze_embeddings(self):
        """Freeze embedding layers."""
        logger.info("Freezing embeddings")
        
        if hasattr(self.model, 'bert') and hasattr(self.model.bert, 'embeddings'):
            for param in self.model.bert.embeddings.parameters():
                param.requires_grad = False
        elif hasattr(self.model, 'embeddings'):
            for param in self.model.embeddings.parameters():
                param.requires_grad = False
        else:
            logger.warning("Could not find embeddings to freeze")
    
    def train(self, 
              train_data: Dict[str, List],
              val_data: Optional[Dict[str, List]] = None,
              output_dir: str = "./models/trained",
              task_type: str = "classification"):
        """
        Train the BERT model.
        
        Args:
            train_data: Training data with 'texts' and optionally 'labels'
            val_data: Validation data with same format
            output_dir: Output directory for trained model
            task_type: Type of task ('classification' or 'language_modeling')
        """
        logger.info("Starting training...")
        
        # Setup model if not already done
        if self.model is None:
            num_labels = len(set(train_data.get('labels', [0, 1]))) if 'labels' in train_data else 2
            self.setup_model_and_tokenizer(task_type, num_labels)
        
        # Create datasets
        max_length = self.config['data']['max_length']
        
        train_dataset = BERTDataset(
            train_data['texts'],
            train_data.get('labels'),
            self.tokenizer,
            max_length
        )
        
        val_dataset = None
        if val_data:
            val_dataset = BERTDataset(
                val_data['texts'],
                val_data.get('labels'),
                self.tokenizer,
                max_length
            )
        
        # Setup training arguments
        training_args = self.create_training_arguments(output_dir)
        
        # Create data collator
        if task_type == "classification":
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        else:
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=True,
                mlm_probability=0.15
            )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics if task_type == "classification" else None,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=self.config['training']['early_stopping']['patience'],
                early_stopping_threshold=self.config['training']['early_stopping']['threshold']
            )] if self.config['training']['early_stopping'].get('patience', 0) > 0 else None
        )
        
        # Initialize wandb if configured
        if self.config['training']['logging'].get('use_wandb', False):
            wandb.init(
                project=self.config['training']['logging'].get('wandb_project', 'bert-mobile'),
                name=f"bert-training-{int(time.time())}",
                config=self.config
            )
        
        # Custom training loop if knowledge distillation is enabled
        if self.teacher_model is not None:
            self.train_with_distillation(trainer, train_dataset, val_dataset, output_dir)
        else:
            # Standard training
            trainer.train()
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training history
        self.save_training_history(output_dir)
        
        logger.info(f"Training completed. Model saved to {output_dir}")
    
    def train_with_distillation(self, trainer, train_dataset, val_dataset, output_dir):
        """Train with knowledge distillation."""
        logger.info("Training with knowledge distillation...")
        
        # Knowledge distillation parameters
        temperature = self.config['training']['mobile_training']['temperature']
        alpha = self.config['training']['mobile_training']['alpha']
        
        # Custom training loop
        train_dataloader = trainer.get_train_dataloader()
        optimizer = trainer.create_optimizer()
        scheduler = trainer.create_scheduler(len(train_dataloader) * trainer.args.num_train_epochs, optimizer)
        
        self.model.train()
        global_step = 0
        
        for epoch in range(int(trainer.args.num_train_epochs)):
            logger.info(f"Epoch {epoch + 1}/{int(trainer.args.num_train_epochs)}")
            
            epoch_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Student forward pass
                student_outputs = self.model(**batch)
                
                # Teacher forward pass
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(**batch)
                
                # Calculate distillation loss
                distillation_loss = self.calculate_distillation_loss(
                    student_outputs, teacher_outputs, batch.get('labels'), temperature, alpha
                )
                
                # Backward pass
                distillation_loss.backward()
                
                # Update weights
                if (step + 1) % trainer.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), trainer.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                epoch_loss += distillation_loss.item()
                
                # Log progress
                if global_step % trainer.args.logging_steps == 0:
                    self.training_history['train_loss'].append(distillation_loss.item())
                    self.training_history['learning_rate'].append(scheduler.get_last_lr()[0])
                    
                    if self.config['training']['logging'].get('use_wandb', False):
                        wandb.log({
                            "train_loss": distillation_loss.item(),
                            "learning_rate": scheduler.get_last_lr()[0],
                            "epoch": epoch,
                            "step": global_step
                        })
            
            # Validation
            if val_dataset and (epoch + 1) % trainer.args.eval_steps == 0:
                val_loss, val_accuracy = self.evaluate_model(val_dataset)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_accuracy'].append(val_accuracy)
                
                if self.config['training']['logging'].get('use_wandb', False):
                    wandb.log({
                        "val_loss": val_loss,
                        "val_accuracy": val_accuracy,
                        "epoch": epoch
                    })
                
                # Save best model
                if val_loss < self.training_state['best_loss']:
                    self.training_state['best_loss'] = val_loss
                    trainer.save_model(os.path.join(output_dir, "best_model"))
            
            logger.info(f"Epoch {epoch + 1} completed. Average loss: {epoch_loss / len(train_dataloader):.4f}")
    
    def calculate_distillation_loss(self, student_outputs, teacher_outputs, labels, temperature, alpha):
        """Calculate knowledge distillation loss."""
        # Hard target loss (if labels available)
        hard_loss = 0
        if labels is not None:
            hard_loss = student_outputs.loss
        
        # Soft target loss
        student_logits = student_outputs.logits
        teacher_logits = teacher_outputs.logits
        
        # Apply temperature
        student_probs = torch.softmax(student_logits / temperature, dim=-1)
        teacher_probs = torch.softmax(teacher_logits / temperature, dim=-1)
        
        # KL divergence loss
        soft_loss = torch.nn.KLDivLoss(reduction='batchmean')(
            torch.log(student_probs), teacher_probs
        ) * (temperature ** 2)
        
        # Combined loss
        if labels is not None:
            total_loss = alpha * hard_loss + (1 - alpha) * soft_loss
        else:
            total_loss = soft_loss
        
        return total_loss
    
    def evaluate_model(self, val_dataset):
        """Evaluate model on validation dataset."""
        self.model.eval()
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                
                total_loss += outputs.loss.item()
                
                if hasattr(outputs, 'logits'):
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    correct_predictions += (predictions == batch['labels']).sum().item()
                    total_predictions += batch['labels'].size(0)
        
        avg_loss = total_loss / len(val_dataloader)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        self.model.train()
        return avg_loss, accuracy
    
    def create_training_arguments(self, output_dir: str) -> TrainingArguments:
        """Create training arguments."""
        training_config = self.config['training']
        
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=training_config['num_epochs'],
            per_device_train_batch_size=training_config['batch_size'],
            per_device_eval_batch_size=training_config['batch_size'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            learning_rate=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
            warmup_ratio=training_config['warmup_ratio'],
            max_grad_norm=training_config['max_grad_norm'],
            fp16=training_config['fp16'],
            bf16=training_config['bf16'],
            dataloader_num_workers=training_config['dataloader_num_workers'],
            save_strategy=training_config['save_strategy'],
            save_steps=training_config['save_steps'],
            save_total_limit=training_config['save_total_limit'],
            evaluation_strategy=training_config['evaluation_strategy'],
            eval_steps=training_config['eval_steps'],
            load_best_model_at_end=training_config['load_best_model_at_end'],
            metric_for_best_model=training_config['metric_for_best_model'],
            logging_steps=training_config['logging_steps'],
            logging_dir=training_config['logging_dir'],
            report_to=training_config['report_to'],
            remove_unused_columns=False,
            push_to_hub=False
        )
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = (predictions == labels).mean()
        
        # Additional metrics
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def save_training_history(self, output_dir: str):
        """Save training history and metrics."""
        history_path = os.path.join(output_dir, "training_history.json")
        
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save training state
        state_path = os.path.join(output_dir, "training_state.json")
        with open(state_path, 'w') as f:
            json.dump(self.training_state, f, indent=2)
        
        # Save configuration used
        config_path = os.path.join(output_dir, "training_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"Training history saved to {history_path}")
    
    def fine_tune(self,
                  train_data: Dict[str, List],
                  val_data: Optional[Dict[str, List]] = None,
                  output_dir: str = "./models/fine_tuned",
                  task_type: str = "classification",
                  num_epochs: Optional[int] = None,
                  learning_rate: Optional[float] = None):
        """
        Fine-tune the model with optional parameter overrides.
        
        Args:
            train_data: Training data
            val_data: Validation data
            output_dir: Output directory
            task_type: Type of task
            num_epochs: Override number of epochs
            learning_rate: Override learning rate
        """
        # Override config if parameters provided
        if num_epochs:
            self.config['training']['num_epochs'] = num_epochs
        if learning_rate:
            self.config['training']['learning_rate'] = learning_rate
        
        # Use smaller learning rate for fine-tuning
        if learning_rate is None:
            self.config['training']['learning_rate'] = self.config['training']['learning_rate'] * 0.1
        
        # Fine-tune (same as train but with different defaults)
        self.train(train_data, val_data, output_dir, task_type)
    
    def predict(self, texts: List[str], batch_size: int = 32) -> List:
        """Make predictions on new texts."""
        if self.model is None:
            raise ValueError("Model not loaded. Call setup_model_and_tokenizer() first.")
        
        self.model.eval()
        predictions = []
        
        # Create dataset
        dataset = BERTDataset(
            texts, 
            None,  # No labels for prediction
            self.tokenizer, 
            self.config['data']['max_length']
        )
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                
                if hasattr(outputs, 'logits'):
                    # Classification
                    preds = torch.argmax(outputs.logits, dim=-1)
                    predictions.extend(preds.cpu().numpy().tolist())
                else:
                    # Other tasks
                    predictions.extend(outputs.last_hidden_state.cpu().numpy())
        
        return predictions
    
    def save_model(self, output_dir: str):
        """Save model and tokenizer."""
        os.makedirs(output_dir, exist_ok=True)
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save additional metadata
        metadata = {
            'model_type': type(self.model).__name__,
            'vocab_size': len(self.tokenizer.vocab),
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'custom_vocab_used': self.custom_vocab_path is not None,
            'training_config': self.config
        }
        
        with open(os.path.join(output_dir, 'model_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {output_dir}")
    
    def load_model(self, model_dir: str, task_type: str = "classification"):
        """Load saved model and tokenizer."""
        logger.info(f"Loading model from {model_dir}")
        
        # Load metadata
        metadata_path = os.path.join(model_dir, 'model_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Model type: {metadata['model_type']}")
            logger.info(f"Vocab size: {metadata['vocab_size']}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Load model
        if task_type == "classification":
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        else:
            self.model = AutoModel.from_pretrained(model_dir)
        
        self.model.to(self.device)
        logger.info("Model loaded successfully")
    
    def get_model_size(self) -> Dict[str, Any]:
        """Get model size information."""
        if self.model is None:
            return {}
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Estimate size
        param_size = total_params * 4  # 4 bytes per float32 parameter
        model_size_mb = param_size / (1024 * 1024)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'vocab_size': len(self.tokenizer.vocab) if self.tokenizer else 0
        }
