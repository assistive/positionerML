#!/usr/bin/env python3
# fastvlm/scripts/train.py

import argparse
import os
import sys
from pathlib import Path
import yaml
import logging
from typing import Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.trainer import FastVLMTrainer, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> TrainingConfig:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Extract training section
    training_config = config_dict.get('training', {})
    
    # Create TrainingConfig object
    config = TrainingConfig(
        model_name=training_config.get('model_name', 'fastvlm-base'),
        model_path=training_config.get('pretrained_model_path'),
        train_data_path=training_config['data']['train_data_path'],
        val_data_path=training_config['data'].get('val_data_path'),
        num_epochs=training_config.get('num_epochs', 3),
        batch_size=training_config.get('batch_size', 16),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 4),
        learning_rate=float(training_config['optimizer'].get('learning_rate', 5e-5)),
        warmup_ratio=float(training_config['scheduler'].get('warmup_ratio', 0.1)),
        weight_decay=float(training_config['optimizer'].get('weight_decay', 0.01)),
        max_grad_norm=float(training_config.get('max_grad_norm', 1.0)),
        use_lora=training_config['lora'].get('enabled', True),
        lora_rank=training_config['lora'].get('rank', 16),
        lora_alpha=training_config['lora'].get('alpha', 32),
        lora_dropout=float(training_config['lora'].get('dropout', 0.1)),
        lora_target_modules=training_config['lora'].get('target_modules'),
        use_qlora=training_config['qlora'].get('enabled', False),
        qlora_bits=training_config['qlora'].get('bits', 4),
        use_gradient_checkpointing=training_config.get('gradient_checkpointing', True),
        mixed_precision=training_config['mixed_precision'].get('dtype', 'fp16') if training_config['mixed_precision'].get('enabled', True) else None,
        distributed=training_config['distributed'].get('enabled', False),
        deepspeed_config=training_config['deepspeed'].get('config_file') if training_config['deepspeed'].get('enabled', False) else None,
        logging_steps=training_config['logging'].get('logging_steps', 10),
        eval_steps=training_config['logging'].get('eval_steps', 500),
        save_steps=training_config['logging'].get('save_steps', 1000),
        save_total_limit=training_config['logging'].get('save_total_limit', 3),
        output_dir=training_config.get('output_dir', './output'),
        resume_from_checkpoint=training_config.get('resume_from_checkpoint'),
        use_wandb=training_config['wandb'].get('enabled', True),
        wandb_project=training_config['wandb'].get('project', 'fastvlm'),
        wandb_run_name=training_config['wandb'].get('name')
    )
    
    return config


def main():
    parser = argparse.ArgumentParser(description="Train FastVLM models")
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        choices=["fastvlm-tiny", "fastvlm-small", "fastvlm-base", "fastvlm-large"],
        help="Model variant to train"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to pretrained model (if continuing training)"
    )
    
    # Data arguments
    parser.add_argument(
        "--train_data",
        type=str,
        default=None,
        help="Path to training data"
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default=None,
        help="Path to validation data"
    )
    
    # Training arguments
    parser.add_argument(
        "--config",
        type=str,
        default="./config/training_config.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints"
    )
    
    # Optimization arguments
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA for parameter-efficient training"
    )
    parser.add_argument(
        "--use_qlora",
        action="store_true",
        help="Use QLoRA with quantization"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        choices=["fp16", "bf16", "none"],
        default=None,
        help="Mixed precision training"
    )
    
    # Distributed training
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed training"
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="Path to DeepSpeed config file"
    )
    
    # Other arguments
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Weights & Biases run name"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.model_name:
        config.model_name = args.model_name
    if args.model_path:
        config.model_path = args.model_path
    if args.train_data:
        config.train_data_path = args.train_data
    if args.val_data:
        config.val_data_path = args.val_data
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.use_lora:
        config.use_lora = True
    if args.use_qlora:
        config.use_qlora = True
    if args.gradient_checkpointing:
        config.use_gradient_checkpointing = True
    if args.mixed_precision:
        config.mixed_precision = args.mixed_precision if args.mixed_precision != "none" else None
    if args.distributed:
        config.distributed = True
    if args.deepspeed:
        config.deepspeed_config = args.deepspeed
    if args.resume_from_checkpoint:
        config.resume_from_checkpoint = args.resume_from_checkpoint
    if args.wandb_project:
        config.wandb_project = args.wandb_project
    if args.wandb_run_name:
        config.wandb_run_name = args.wandb_run_name
    
    # Set random seed
    import torch
    import numpy as np
    import random
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Log configuration
    logger.info("Training Configuration:")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Train data: {config.train_data_path}")
    logger.info(f"  Val data: {config.val_data_path}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Epochs: {config.num_epochs}")
    logger.info(f"  Output: {config.output_dir}")
    logger.info(f"  LoRA: {config.use_lora}")
    logger.info(f"  QLoRA: {config.use_qlora}")
    logger.info(f"  Mixed precision: {config.mixed_precision}")
    logger.info(f"  Distributed: {config.distributed}")
    
    # Create trainer
    trainer = FastVLMTrainer(config)
    
    # Resume from checkpoint if specified
    if config.resume_from_checkpoint:
        state = trainer.resume_from_checkpoint(config.resume_from_checkpoint)
        if state:
            logger.info(f"Resumed from checkpoint: epoch {state['epoch']}, step {state['global_step']}")
    
    # Start training
    try:
        trainer.train()
        logger.info("Training completed successfully!")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
