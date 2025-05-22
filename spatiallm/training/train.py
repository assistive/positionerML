#!/usr/bin/env python3
"""
Main training script for spatialLM models.

This script handles the complete training process for spatialLM models, including
data preparation, model initialization, training, evaluation, and model saving.
"""

import os
import sys
import logging
import argparse
import json
import torch
import numpy as np
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import (
    AutoTokenizer,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    set_seed
)

from models.spatialLM import SpatialLM, SpatialLMConfig
from utils.dataset import load_and_prepare_dataset, load_spatial_dataset, create_augmented_spatial_dataset
from utils.metrics import compute_metrics, compute_spatial_metrics, evaluate_spatial_language_understanding

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("train")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train a spatialLM model")
    
    # Model arguments
    parser.add_argument(
        "--base_model",
        type=str,
        default="gpt2",
        help="Base model name or path (e.g., gpt2, gpt2-medium, etc.)"
    )
    
    parser.add_argument(
        "--model_type",
        type=str,
        default="gpt2",
        help="Model type (e.g., gpt2, gpt_neo, llama, etc.)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./trained_models",
        help="Directory to save the trained model"
    )
    
    parser.add_argument(
        "--from_pretrained",
        type=str,
        help="Path to a pretrained spatialLM model to continue training"
    )
    
    # Spatial configuration
    parser.add_argument(
        "--spatial_dim",
        type=int,
        default=3,
        help="Number of spatial dimensions (typically 3 for x, y, z)"
    )
    
    parser.add_argument(
        "--spatial_hidden_size",
        type=int,
        default=256,
        help="Size of hidden layers in spatial components"
    )
    
    parser.add_argument(
        "--spatial_dropout",
        type=float,
        default=0.1,
        help="Dropout rate for spatial layers"
    )
    
    parser.add_argument(
        "--use_spatial_attention",
        action="store_true",
        help="Whether to use spatial attention mechanism"
    )
    
    parser.add_argument(
        "--spatial_attention_heads",
        type=int,
        default=4,
        help="Number of attention heads in spatial attention"
    )
    
    # Data arguments
    parser.add_argument(
        "--train_file",
        type=str,
        help="Path to training data file (csv, json, txt, or jsonl)"
    )
    
    parser.add_argument(
        "--validation_file",
        type=str,
        help="Path to validation data file (csv, json, txt, or jsonl)"
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Dataset name if using datasets from Hugging Face Hub"
    )
    
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        help="Dataset configuration name if using datasets from Hugging Face Hub"
    )
    
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Column in the dataset containing the text"
    )
    
    parser.add_argument(
        "--spatial_mode",
        action="store_true",
        help="Whether to train in spatial mode (with coordinate data)"
    )
    
    parser.add_argument(
        "--coordinate_columns",
        type=str,
        default="x,y,z",
        help="Comma-separated list of column names containing spatial coordinates"
    )
    
    parser.add_argument(
        "--data_augmentation",
        action="store_true",
        help="Whether to apply data augmentation for spatial data"
    )
    
    parser.add_argument(
        "--augmentation_factor",
        type=int,
        default=2,
        help="How many augmented copies to create for each example"
    )
    
    # Training arguments
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size per GPU/TPU core/CPU for training"
    )
    
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size per GPU/TPU core/CPU for evaluation"
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period)"
    )
    
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay to apply (if not zero) to all layers except bias/LayerNorm weights"
    )
    
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm (for gradient clipping)"
    )
    
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=3.0,
        help="Total number of training epochs to perform"
    )
    
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="If > 0: set total number of training steps to perform. Overrides num_train_epochs"
    )
    
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.0,
        help="Ratio of total training steps used for warmup"
    )
    
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=500,
        help="Log every X updates steps"
    )
    
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Run evaluation every X steps"
    )
    
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps"
    )
    
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
        help="Limit the total amount of checkpoints, delete the older checkpoints"
    )
    
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training"
    )
    
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to use bf16 (mixed) precision instead of 32-bit"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization"
    )
    
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub"
    )
    
    parser.add_argument(
        "--hub_model_id",
        type=str,
        help="The name of the repository to keep in sync with the local `output_dir`"
    )
    
    parser.add_argument(
        "--hub_token",
        type=str,
        help="The token to use to push to the Model Hub"
    )
    
    # Early stopping
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        help="Number of evaluation rounds with no improvement after which training will be stopped"
    )
    
    parser.add_argument(
        "--early_stopping_threshold",
        type=float,
        default=0.0,
        help="Minimum improvement required to count as improvement"
    )
    
    return parser.parse_args()

def create_model(args):
    """
    Create or load a spatialLM model.
    
    Args:
        args: Command line arguments
        
    Returns:
        model: SpatialLM model
        tokenizer: Tokenizer for the model
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if args.from_pretrained:
        # Load existing spatialLM model
        logger.info(f"Loading pretrained spatialLM model from {args.from_pretrained}")
        model = SpatialLM.from_pretrained(args.from_pretrained)
    else:
        # Create a new spatialLM model
        logger.info(f"Creating new spatialLM model based on {args.base_model}")
        
        # Create configuration
        config = SpatialLMConfig(
            base_model_type=args.model_type,
            base_model_name=args.base_model,
            spatial_dim=args.spatial_dim,
            spatial_hidden_size=args.spatial_hidden_size,
            spatial_dropout=args.spatial_dropout,
            use_spatial_attention=args.use_spatial_attention,
            spatial_attention_heads=args.spatial_attention_heads
        )
        
        # Create model
        model = SpatialLM(config)
    
    return model, tokenizer

def create_training_args(args):
    """
    Create training arguments for the Trainer.
    
    Args:
        args: Command line arguments
        
    Returns:
        training_args: TrainingArguments object
    """
    # Create output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(args.base_model)
    output_dir = os.path.join(args.output_dir, f"{model_name}_{timestamp}")
    
    # Create TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        
        # Training hyperparameters
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        
        # Logging & Evaluation
        logging_dir=os.path.join(output_dir, "logs"),
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        
        # Precision
        fp16=args.fp16,
        bf16=args.bf16,
        
        # Misc
        seed=args.seed,
        data_seed=args.seed,
        
        # Hugging Face Hub
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_token=args.hub_token,
        
        # Report training metrics to mlflow/wandb/tensorboard
        report_to=["tensorboard"],
    )
    
    return training_args

def save_training_config(args, output_dir):
    """Save training configuration to a JSON file"""
    config = vars(args)
    config["training_date"] = datetime.datetime.now().isoformat()
    
    # Save config
    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Training configuration saved to {config_path}")

def train(args):
    """Main training function"""
    # Set random seed
    set_seed(args.seed)
    
    # Create or load model
    model, tokenizer = create_model(args)
    
    # Load dataset
    if args.spatial_mode:
        # Parse coordinate columns
        coordinate_columns = args.coordinate_columns.split(",")
        
        # Load spatial dataset
        dataset = load_spatial_dataset(
            tokenizer=tokenizer,
            spatial_file=args.train_file,
            coordinate_columns=coordinate_columns,
            text_column=args.text_column,
            validation_split_percentage=10,
            test_split_percentage=None,
        )
        
        # Apply data augmentation if requested
        if args.data_augmentation and "train" in dataset:
            logger.info(f"Applying spatial data augmentation with factor {args.augmentation_factor}")
            dataset["train"] = create_augmented_spatial_dataset(
                dataset=dataset["train"],
                augmentation_factor=args.augmentation_factor,
                noise_std=0.1,
                coordinate_columns=[f"coord_{col}" for col in coordinate_columns],
            )
            
        # Create a simple data collator that will also handle the coordinate data
        from dataclasses import dataclass
        
        @dataclass
        class SpatialDataCollator:
            """Data collator for spatial data"""
            
            def __call__(self, features):
                """Collate spatial features"""
                batch = {}
                
                # Collate standard fields
                for key in features[0].keys():
                    if key.startswith("coord_"):
                        # For coordinate fields, simply stack the values
                        batch[key.replace("coord_", "")] = torch.tensor(
                            [f[key] for f in features], dtype=torch.float32
                        )
                    elif key in ["input_ids", "attention_mask", "token_type_ids", "labels"]:
                        # For standard fields, apply padding
                        batch[key] = torch.tensor([f[key] for f in features], dtype=torch.long)
                
                # Create spatial_coordinates tensor
                if all(f"coord_{col}" in features[0] for col in coordinate_columns):
                    batch["spatial_coordinates"] = torch.tensor(
                        [[f[f"coord_{col}"] for col in coordinate_columns] for f in features],
                        dtype=torch.float32
                    )
                
                return batch
        
        data_collator = SpatialDataCollator()
    else:
        # Load standard language modeling dataset
        dataset = load_and_prepare_dataset(
            tokenizer=tokenizer,
            dataset_name=args.dataset_name,
            dataset_config_name=args.dataset_config_name,
            train_file=args.train_file,
            validation_file=args.validation_file,
            text_column=args.text_column,
        )
        
        # Create data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # Causal language modeling, not masked
        )
    
    # Create training arguments
    training_args = create_training_args(args)
    
    # Create early stopping callback
    callbacks = []
    if args.early_stopping_patience > 0:
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold
        )
        callbacks.append(early_stopping_callback)
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if "validation" in dataset else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )
    
    # Save training configuration
    save_training_config(args, training_args.output_dir)
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    logger.info(f"Saving trained model to {training_args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    # Push to Hub if requested
    if args.push_to_hub:
        logger.info("Pushing model to Hugging Face Hub...")
        trainer.push_to_hub()
    
    logger.info("Training completed!")
    return training_args.output_dir

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Perform training
    output_dir = train(args)
    
    logger.info(f"Training completed. Model saved to {output_dir}")

if __name__ == "__main__":
    main()
