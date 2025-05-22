# internvl/scripts/fine_tune.py

#!/usr/bin/env python3

import argparse
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from fine_tuner import InternVLFineTuner
from data_processor import DataProcessor
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description='Fine-tune InternVL model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to pretrained model')
    parser.add_argument('--train_data', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--val_data', type=str,
                       help='Path to validation data')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Path to training configuration')
    parser.add_argument('--output_dir', type=str, default='./models/fine_tuned',
                       help='Output directory for fine-tuned model')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Training batch size')
    parser.add_argument('--num_epochs', type=int,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--learning_rate', type=float,
                       help='Learning rate (overrides config)')
    parser.add_argument('--use_lora', action='store_true',
                       help='Enable LoRA fine-tuning')
    parser.add_argument('--use_qlora', action='store_true',
                       help='Enable QLoRA fine-tuning')
    
    args = parser.parse_args()
    
    try:
        print("Initializing fine-tuner...")
        
        # Initialize fine-tuner
        fine_tuner = InternVLFineTuner(args.model_path, args.config)
        
        # Override config with command line arguments
        if args.num_epochs:
            fine_tuner.config['training']['num_epochs'] = args.num_epochs
        if args.learning_rate:
            fine_tuner.config['training']['learning_rate'] = args.learning_rate
        if args.use_lora:
            fine_tuner.config['training']['lora']['enabled'] = True
        if args.use_qlora:
            fine_tuner.config['training']['qlora']['enabled'] = True
        
        # Setup model and tokenizer
        fine_tuner.setup_model_and_tokenizer()
        
        print("Setting up data loaders...")
        
        # Initialize data processor
        data_processor = DataProcessor(
            fine_tuner.tokenizer,
            fine_tuner.config['training']['data']
        )
        
        # Create data loaders
        dataloaders = data_processor.create_dataloaders(
            train_path=args.train_data,
            val_path=args.val_data,
            batch_size=args.batch_size
        )
        
        print("Starting fine-tuning...")
        
        # Start fine-tuning
        fine_tuner.fine_tune(
            train_dataloader=dataloaders['train'],
            val_dataloader=dataloaders.get('val'),
            output_dir=args.output_dir
        )
        
        print("Fine-tuning completed successfully!")
        
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

