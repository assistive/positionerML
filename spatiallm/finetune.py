#!/usr/bin/env python3
"""
finetune.py

Script for fine-tuning spatialLM models on custom datasets.
Supports various fine-tuning techniques including:
- Full fine-tuning
- LoRA (Low-Rank Adaptation)
- QLoRA (Quantized Low-Rank Adaptation)
- Parameter-Efficient Fine-Tuning (PEFT)
"""

import os
import sys
import logging
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Union, Tuple

from transformers import (
    Trainer, TrainingArguments, 
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM, AutoTokenizer,
    set_seed, get_scheduler,
    EarlyStoppingCallback
)

from peft import (
    get_peft_model, LoraConfig, 
    TaskType, PeftModel, PeftConfig
)

from datasets import load_dataset, Dataset, DatasetDict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dataset import load_and_prepare_dataset
from utils.metrics import compute_metrics
from models.optimization import create_optimizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("finetune")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Fine-tune spatialLM models")
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to pre-trained model or model identifier from huggingface.co/models"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./finetuned_models",
        help="Directory to save the fine-tuned model"
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
        "--train_split",
        type=str,
        default="train",
        help="Dataset split to use for training"
    )
    
    parser.add_argument(
        "--validation_split",
        type=str,
        default="validation",
        help="Dataset split to use for validation"
    )
    
    # Training arguments
    parser.add_argument(
        "--finetune_method",
        type=str,
        default="full",
        choices=["full", "lora", "qlora", "peft"],
        help="Fine-tuning method to use"
    )
    
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
    
    # LoRA specific arguments
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="Rank of LoRA matrices"
    )
    
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="Alpha parameter for LoRA scaling"
    )
    
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="Dropout probability for LoRA layers"
    )
    
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,v_proj",
        help="Comma-separated list of module names to apply LoRA to"
    )
    
    # QLora specific arguments
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="Number of bits to quantize to"
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

def setup_model_and_tokenizer(args):
    """
    Load model and tokenizer based on fine-tuning method
    
    Args:
        args: Command line arguments
        
    Returns:
        model: The model to fine-tune
        tokenizer: The tokenizer
    """
    logger.info(f"Loading model and tokenizer from {args.model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model based on fine-tuning method
    if args.finetune_method == "full":
        # Standard full fine-tuning
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
        )
    
    elif args.finetune_method == "lora":
        # LoRA fine-tuning
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
        )
        
        # Parse target modules
        target_modules = args.lora_target_modules.split(",")
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Create LoRA model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    elif args.finetune_method == "qlora":
        # QLoRA fine-tuning (quantized LoRA)
        from transformers import BitsAndBytesConfig
        
        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Load quantized model
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # Parse target modules
        target_modules = args.lora_target_modules.split(",")
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Create LoRA model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    elif args.finetune_method == "peft":
        # Other PEFT methods (adapter-based fine-tuning)
        # This is a placeholder - you can implement other PEFT methods as needed
        raise NotImplementedError("Other PEFT methods are not implemented yet")
    
    return model, tokenizer

def create_training_args(args):
    """
    Create training arguments for the Trainer
    
    Args:
        args: Command line arguments
        
    Returns:
        training_args: TrainingArguments object
    """
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(args.model_path)
    output_dir = os.path.join(args.output_dir, f"{model_name}_{args.finetune_method}_{timestamp}")
    
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
    config["training_date"] = datetime.now().isoformat()
    
    # Save config
    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Training configuration saved to {config_path}")

def finetune(args):
    """Main fine-tuning function"""
    # Set random seed
    set_seed(args.seed)
    
    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args)
    
    # Load and prepare dataset
    dataset = load_and_prepare_dataset(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        train_file=args.train_file,
        validation_file=args.validation_file,
        text_column=args.text_column,
        train_split=args.train_split,
        validation_split=args.validation_split,
        max_seq_length=tokenizer.model_max_length,
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're fine-tuning a causal language model, not a masked language model
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
    logger.info("Starting fine-tuning...")
    trainer.train()
    
    # Save the final model
    logger.info(f"Saving fine-tuned model to {training_args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    # Push to Hub if requested
    if args.push_to_hub:
        logger.info("Pushing model to Hugging Face Hub...")
        trainer.push_to_hub()
    
    logger.info("Fine-tuning completed!")
    return training_args.output_dir

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Perform fine-tuning
    output_dir = finetune(args)
    
    logger.info(f"Fine-tuning completed. Model saved to {output_dir}")

if __name__ == "__main__":
    main()
