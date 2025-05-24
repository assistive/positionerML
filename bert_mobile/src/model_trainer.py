# src/bert_trainer.py

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple
import logging
import yaml
from pathlib import Path
import numpy as np
from tqdm import tqdm
import time

from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    BertTokenizer, BertModel, BertConfig,
    Trainer, TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training state
        self.training_state = {
            'epoch': 0,
            'step': 0,
            'best_loss': float('inf'),
            'best_accuracy': 0.0
        }
    
    def load_config(self, config_path: str) -> Dict:
        """Load training configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_model_and_tokenizer(self):
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
        
        # Load model
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
    
    def setup_knowledge_distillation(self):
        """Setup knowledge distillation from a teacher model."""
        teacher_model_name = self.config['training']['mobile_training']['teacher_model']
        
        logger.info(f"Setting up knowledge distillation with teacher: {teacher_model_name}")
        
        # Load teacher model
        self.teacher_model = AutoModel.from_pretrained(teacher_model_name)
        self.teacher_model.to(self.device)
        self.teacher_model.eval()
        
        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def reduce_model_layers(self):
        """Reduce the number of layers in the model for mobile optimization."""
        mobile_layers = self.config['training']['mobile_training'].get('num_layers', 6)
        
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            original_layers = len(self.model.encoder.layer)
            if mobile_layers < original_layers:
                logger.info(f"Reducing layers from {original_layers} to {mobile_layers}")
                self.model.encoder.layer = self.model.encoder.layer[:mobile_layers]
                self.model.config.num_hidden_layers = mobile_layers
    
    def freeze_encoder_layers(self, num_layers: int):
        """Freeze the first N encoder layers."""
        logger.info(f"Freezing first {num_layers} encoder layers")
        
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            for i in range(min(num_layers,
