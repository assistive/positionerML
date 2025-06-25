# internvl/scripts/train_model.py

#!/usr/bin/env python3

import argparse
import sys
import torch
from pathlib import Path

# Add src to path  
sys.path.append(str(Path(__file__).parent.parent / "src"))

from trainer import InternVLTrainer
from data_processor import DataProcessor
from model_downloader import ModelDownloader

def main():
    parser = argparse.ArgumentParser(description='Train InternVL model from scratch')
    parser.add_argument('--model_name', type=str, default='internvl2-2b',
                       help='Model name to download and train')
    parser.add_argument('--train_data', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--val_data', type=str,
                       help='Path to validation data')
    parser.add_argument('--output_dir', type=str, default='./models/trained',
                       help='Output directory for trained model')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Path to training configuration')
    parser.add_argument('--download_model', action='store_true',
                       help='Download model before training')
    parser.add_argument('--resume_from_checkpoint', type=str,
                       help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    try:
        model_path = None
        
        # Download model if requested
        if args.download_model:
            print("Downloading model...")
            downloader = ModelDownloader()
            model_path, _ = downloader.download_model(args.model_name)
            print(f"Model downloaded to: {model_path}")
        else:
            # Assume model is already available
            model_path = f"./models/pretrained/{args.model_name}"
            if not Path(model_path).exists():
                print(f"Model not found at {model_path}. Use --download_model to download it first.")
                sys.exit(1)
        
        print("Initializing trainer...")
        
        # Initialize trainer
        trainer = InternVLTrainer(
            model_path=model_path,
            config_path=args.config
        )
        
        # Setup model and tokenizer
        trainer.setup_model_and_tokenizer()
        
        print("Setting up data loaders...")
        
        # Initialize data processor
        data_processor = DataProcessor(
            trainer.tokenizer,
            trainer.config['training']['data']
        )
        
        # Create data loaders
        dataloaders = data_processor.create_dataloaders(
            train_path=args.train_data,
            val_path=args.val_data,
            batch_size=trainer.config['training']['batch_size']
        )
        
        print("Starting training...")
        
        # Start training
        trainer.train(
            train_dataloader=dataloaders['train'],
            val_dataloader=dataloaders.get('val'),
            output_dir=args.output_dir,
            resume_from_checkpoint=args.resume_from_checkpoint
        )
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

