# bert_mobile/scripts/fine_tune_bert.py

#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model_trainer import BERTMobileTrainer

def main():
    parser = argparse.ArgumentParser(description='Fine-tune BERT model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to base BERT model')
    parser.add_argument('--train_data', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--val_data', type=str,
                       help='Path to validation data')
    parser.add_argument('--custom_vocab', type=str,
                       help='Path to custom vocabulary')
    parser.add_argument('--output_dir', type=str, default='./models/fine_tuned',
                       help='Output directory for fine-tuned model')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Path to training configuration')
    parser.add_argument('--task_type', type=str, choices=['classification', 'language_modeling'],
                       default='classification', help='Type of fine-tuning task')
    parser.add_argument('--num_labels', type=int, default=2,
                       help='Number of labels for classification')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of fine-tuning epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate for fine-tuning')
    parser.add_argument('--enable_distillation', action='store_true',
                       help='Enable knowledge distillation')
    parser.add_argument('--teacher_model', type=str,
                       help='Path to teacher model for distillation')
    parser.add_argument('--freeze_layers', type=int, default=0,
                       help='Number of encoder layers to freeze')
    
    args = parser.parse_args()
    
    try:
        print("Initializing BERT fine-tuner...")
        
        # Initialize trainer
        trainer = BERTMobileTrainer(
            model_path=args.model_path,
            config_path=args.config,
            custom_vocab_path=args.custom_vocab
        )
        
        # Configure fine-tuning settings
        if args.enable_distillation:
            trainer.config['training']['mobile_training']['knowledge_distillation'] = True
            if args.teacher_model:
                trainer.config['training']['mobile_training']['teacher_model'] = args.teacher_model
        
        if args.freeze_layers > 0:
            trainer.config['training']['fine_tuning']['freeze_encoder_layers'] = args.freeze_layers
        
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
        
        # Start fine-tuning
        trainer.fine_tune(
            train_data=train_dict,
            val_data=val_dict,
            output_dir=args.output_dir,
            task_type=args.task_type,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate
        )
        
        # Print model statistics
        model_stats = trainer.get_model_size()
        print(f"\nFine-tuning completed successfully!")
        print(f"Model statistics:")
        for key, value in model_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value:,}")
        
        # Test the fine-tuned model
        print(f"\nTesting fine-tuned model...")
        test_texts = ["This is a test sentence.", "Another test for the model."]
        predictions = trainer.predict(test_texts)
        print(f"Test predictions: {predictions}")
        
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

