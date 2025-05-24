# bert_mobile/scripts/train_bert.py

#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model_trainer import BERTMobileTrainer

def main():
    parser = argparse.ArgumentParser(description='Train BERT model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to base BERT model')
    parser.add_argument('--train_data', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--val_data', type=str,
                       help='Path to validation data')
    parser.add_argument('--custom_vocab', type=str,
                       help='Path to custom vocabulary')
    parser.add_argument('--output_dir', type=str, default='./models/trained',
                       help='Output directory for trained model')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Path to training configuration')
    parser.add_argument('--task_type', type=str, choices=['classification', 'language_modeling'],
                       default='classification', help='Type of training task')
    parser.add_argument('--num_labels', type=int, default=2,
                       help='Number of labels for classification')
    parser.add_argument('--num_epochs', type=int,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--learning_rate', type=float,
                       help='Learning rate (overrides config)')
    parser.add_argument('--batch_size', type=int,
                       help='Batch size (overrides config)')
    parser.add_argument('--enable_mobile_optimizations', action='store_true',
                       help='Enable mobile-specific optimizations')
    
    args = parser.parse_args()
    
    try:
        print("Initializing BERT trainer...")
        
        # Initialize trainer
        trainer = BERTMobileTrainer(
            model_path=args.model_path,
            config_path=args.config,
            custom_vocab_path=args.custom_vocab
        )
        
        # Override config with command line arguments
        if args.num_epochs:
            trainer.config['training']['num_epochs'] = args.num_epochs
        if args.learning_rate:
            trainer.config['training']['learning_rate'] = args.learning_rate
        if args.batch_size:
            trainer.config['training']['batch_size'] = args.batch_size
        
        # Enable mobile optimizations
        if args.enable_mobile_optimizations:
            trainer.config['training']['mobile_training']['enabled'] = True
        
        # Setup model and tokenizer
        trainer.setup_model_and_tokenizer(args.task_type, args.num_labels)
        
        # Load training data
        import json
        with open(args.train_data, 'r') as f:
            train_data = json.load(f)
        
        val_data = None
        if args.val_data:
            with open(args.val_data, 'r') as f:
                val_data = json.load(f)
        
        # Prepare data format
        if args.task_type == 'classification':
            train_dict = {
                'texts': [item['text'] for item in train_data],
                'labels': [item['label'] for item in train_data]
            }
            
            val_dict = None
            if val_data:
                val_dict = {
                    'texts': [item['text'] for item in val_data],
                    'labels': [item['label'] for item in val_data]
                }
        else:
            train_dict = {
                'texts': [item['text'] for item in train_data]
            }
            
            val_dict = None
            if val_data:
                val_dict = {
                    'texts': [item['text'] for item in val_data]
                }
        
        # Start training
        trainer.train(
            train_data=train_dict,
            val_data=val_dict,
            output_dir=args.output_dir,
            task_type=args.task_type
        )
        
        # Print model statistics
        model_stats = trainer.get_model_size()
        print(f"\nTraining completed successfully!")
        print(f"Model statistics:")
        for key, value in model_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value:,}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
