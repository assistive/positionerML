# bert_mobile/scripts/build_vocabulary.py

#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from vocab_builder import VocabularyBuilder

def main():
    parser = argparse.ArgumentParser(description='Build custom vocabulary for BERT')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing text files')
    parser.add_argument('--output_dir', type=str, default='./data/vocabularies',
                       help='Output directory for vocabulary files')
    parser.add_argument('--vocab_size', type=int, default=30000,
                       help='Target vocabulary size')
    parser.add_argument('--min_frequency', type=int, default=5,
                       help='Minimum token frequency')
    parser.add_argument('--vocab_name', type=str, default='custom_vocab',
                       help='Name for the vocabulary')
    parser.add_argument('--config', type=str, default='config/vocab_config.yaml',
                       help='Path to vocabulary configuration')
    parser.add_argument('--analyze_domain', action='store_true',
                       help='Analyze domain-specific terms')
    
    args = parser.parse_args()
    
    try:
        print(f"Classification data processed and saved to {output_path}")
        print(f"Label mapping: {label_map}")
        return str(output_path)

def main():
    parser = argparse.ArgumentParser(description='Prepare data for BERT training')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory or CSV file')
    parser.add_argument('--vocab_path', type=str, required=True,
                       help='Path to vocabulary/tokenizer')
    parser.add_argument('--output_dir', type=str, default='./data/processed',
                       help='Output directory for processed data')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--task_type', type=str, choices=['classification', 'language_modeling'],
                       default='language_modeling', help='Type of task')
    parser.add_argument('--text_column', type=str, default='text',
                       help='Name of text column (for classification)')
    parser.add_argument('--label_column', type=str, default='label',
                       help='Name of label column (for classification)')
    
    args = parser.parse_args()
    
    try:
        print("Processing data for BERT training...")
        
        # Initialize processor
        processor = BERTDataProcessor(args.vocab_path, args.max_length)
        
        # Process based on task type
        if args.task_type == 'classification':
            if not args.input_dir.endswith('.csv'):
                raise ValueError("Classification task requires CSV input file")
            
            output_path = processor.process_classification_data(
                args.input_dir, args.output_dir, 
                args.text_column, args.label_column
            )
        else:
            output_path = processor.process_text_files(
                args.input_dir, args.output_dir
            )
        
        print("Data preparation completed successfully!")
        print(f"Processed data saved to: {output_path}")
        
    except Exception as e:
        print(f"Error preparing data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

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
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

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
    
    args = parser.parse_args()
    
    try:
        print("Initializing BERT fine-tuner...")
        
        # Initialize trainer
        trainer = BERTMobileTrainer(
            model_path=args.model_path,
            config_path=args.config,
            custom_vocab_path=args.custom_vocab
        )
        
        # Enable mobile training optimizations for fine-tuning
        if args.enable_distillation:
            trainer.config['training']['mobile_training']['knowledge_distillation'] = True
        
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
        
        print("Fine-tuning completed successfully!")
        
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# bert_mobile/scripts/convert_to_mobile.py

#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from mobile_converter import BERTMobileConverter

def main():
    parser = argparse.ArgumentParser(description='Convert BERT model for mobile deployment')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained BERT model')
    parser.add_argument('--output_dir', type=str, default='./models/mobile',
                       help='Output directory for mobile models')
    parser.add_argument('--platforms', type=str, nargs='+', 
                       choices=['ios', 'android', 'both'], default=['both'],
                       help='Target platforms')
    parser.add_argument('--sequence_length', type=int, default=128,
                       help='Maximum sequence length for mobile')
    parser.add_argument('--task_type', type=str, choices=['classification', 'feature_extraction'],
                       default='classification', help='Task type')
    parser.add_argument('--quantize', action='store_true', default=True,
                       help='Apply quantization')
    parser.add_argument('--optimize', action='store_true',
                       help='Apply mobile optimizations')
    parser.add_argument('--validate', action='store_true',
                       help='Validate converted models')
    parser.add_argument('--config', type=str, default='config/mobile_config.yaml',
                       help='Path to mobile configuration')
    
    args = parser.parse_args()
    
    # Expand 'both' platform
    platforms = []
    for platform in args.platforms:
        if platform == 'both':
            platforms.extend(['ios', 'android'])
        else:
            platforms.append(platform)
    
    # Remove duplicates
    platforms = list(set(platforms))
    
    try:
        print("Initializing mobile converter...")
        
        # Initialize converter
        converter = BERTMobileConverter(args.config)
        
        # Apply optimizations if requested
        model_path = args.model_path
        if args.optimize:
            print("Applying mobile optimizations...")
            model_path = converter.optimize_for_mobile(
                args.model_path, 
                converter.config['mobile']['optimization']
            )
        
        # Convert for each platform
        converted_models = {}
        
        for platform in platforms:
            print(f"\\nConverting model for {platform.upper()}...")
            
            platform_dir = Path(args.output_dir) / platform
            platform_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                if platform == 'ios':
                    converted_path = converter.convert_to_coreml(
                        model_path=model_path,
                        output_path=str(platform_dir),
                        sequence_length=args.sequence_length,
                        task_type=args.task_type
                    )
                    
                elif platform == 'android':
                    converted_path = converter.convert_to_tflite(
                        model_path=model_path,
                        output_path=str(platform_dir),
                        sequence_length=args.sequence_length,
                        task_type=args.task_type
                    )
                
                converted_models[platform] = converted_path
                
                # Get model size info
                size_info = converter.get_model_size(converted_path)
                print(f"  Converted model size: {size_info['size_mb']:.2f} MB")
                
                # Validate conversion if requested
                if args.validate:
                    print(f"  Validating {platform} conversion...")
                    validation_results = converter.validate_conversion(
                        original_model_path=args.model_path,
                        converted_model_path=converted_path,
                        platform=platform
                    )
                    
                    if validation_results['conversion_successful']:
                        print(f"  ✓ Validation passed (MAE: {validation_results['mae']:.6f})")
                    else:
                        print(f"  ✗ Validation failed: {validation_results.get('error', 'Unknown error')}")
                
                # Benchmark performance
                print(f"  Benchmarking {platform} model...")
                benchmark_results = converter.benchmark_model(converted_path, platform)
                if benchmark_results:
                    print(f"  Average inference time: {benchmark_results['avg_inference_time_ms']:.1f}ms")
                    print(f"  Throughput: {benchmark_results['fps']:.2f} FPS")
                
            except Exception as e:
                print(f"  Error converting for {platform}: {e}")
                continue
        
        # Summary
        print(f"\\nConversion Summary:")
        print(f"Original model: {args.model_path}")
        print(f"Output directory: {args.output_dir}")
        
        for platform, path in converted_models.items():
            print(f"  {platform.upper()}: {path}")
        
        print("\\nMobile conversion completed successfully!")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# bert_mobile/scripts/evaluate_model.py

#!/usr/bin/env python3

import argparse
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model_trainer import BERTMobileTrainer

# Mobile imports
try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

class BERTEvaluator:
    """Evaluate BERT models on various metrics."""
    
    def __init__(self):
        pass
    
    def evaluate_pytorch_model(self, model_path: str, test_data_path: str, 
                              task_type: str = 'classification') -> Dict[str, Any]:
        """Evaluate PyTorch BERT model."""
        print("Evaluating PyTorch model...")
        
        # Load model
        trainer = BERTMobileTrainer(model_path)
        trainer.load_model(model_path, task_type)
        
        # Load test data
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        
        texts = [item['text'] for item in test_data]
        
        if task_type == 'classification':
            true_labels = [item['label'] for item in test_data]
            
            # Make predictions
            start_time = time.time()
            predictions = trainer.predict(texts)
            inference_time = time.time() - start_time
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            
            accuracy = accuracy_score(true_labels, predictions)
            report = classification_report(true_labels, predictions, output_dict=True)
            cm = confusion_matrix(true_labels, predictions)
            
            results = {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'inference_time_total': inference_time,
                'inference_time_per_sample': inference_time / len(texts),
                'samples_per_second': len(texts) / inference_time
            }
            
        else:
            # For language modeling, calculate perplexity
            results = {
                'perplexity': 'Not implemented',
                'inference_time_total': 0,
                'inference_time_per_sample': 0,
                'samples_per_second': 0
            }
        
        return results
    
    def evaluate_coreml_model(self, model_path: str, test_data_path: str) -> Dict[str, Any]:
        """Evaluate CoreML model."""
        if not COREML_AVAILABLE:
            return {'error': 'CoreML not available'}
        
        print("Evaluating CoreML model...")
        
        # Load model
        model = ct.models.model.MLModel(model_path)
        
        # Load test data
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        
        # Simple tokenization for testing
        def simple_tokenize(text, max_length=128):
            tokens = text.lower().split()[:max_length-2]  # Reserve space for CLS and SEP
            input_ids = [101] + [hash(token) % 30000 for token in tokens] + [102]  # CLS + tokens + SEP
            input_ids += [0] * (max_length - len(input_ids))  # Pad
            attention_mask = [1] * (len(tokens) + 2) + [0] * (max_length - len(tokens) - 2)
            return input_ids[:max_length], attention_mask[:max_length]
        
        predictions = []
        inference_times = []
        
        for item in test_data[:100]:  # Test on subset for speed
            input_ids, attention_mask = simple_tokenize(item['text'])
            
            # Convert to MLMultiArray
            input_ids_array = ct.models.utils.MLMultiArray(
                np.array(input_ids, dtype=np.int32).reshape(1, -1)
            )
            attention_mask_array = ct.models.utils.MLMultiArray(
                np.array(attention_mask, dtype=np.int32).reshape(1, -1)
            )
            
            # Run inference
            start_time = time.time()
            try:
                output = model.predict({
                    'input_ids': input_ids_array,
                    'attention_mask': attention_mask_array
                })
                inference_times.append(time.time() - start_time)
                
                # Get prediction (assuming classification)
                logits = list(output.values())[0]
                pred = np.argmax(logits)
                predictions.append(pred)
                
            except Exception as e:
                print(f"Error in prediction: {e}")
                continue
        
        # Calculate metrics
        if len(predictions) > 0:
            avg_inference_time = np.mean(inference_times)
            results = {
                'avg_inference_time': avg_inference_time,
                'min_inference_time': np.min(inference_times),
                'max_inference_time': np.max(inference_times),
                'samples_per_second': 1.0 / avg_inference_time,
                'successful_predictions': len(predictions),
                'total_attempts': len(test_data[:100])
            }
        else:
            results = {'error': 'No successful predictions'}
        
        return results
    
    def evaluate_tflite_model(self, model_path: str, test_data_path: str) -> Dict[str, Any]:
        """Evaluate TensorFlow Lite model."""
        if not TF_AVAILABLE:
            return {'error': 'TensorFlow not available'}
            
        print("Evaluating TFLite model...")
        
        # Load model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Load test data
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        
        # Simple tokenization for testing
        def simple_tokenize(text, max_length=128):
            tokens = text.lower().split()[:max_length-2]
            input_ids = [101] + [hash(token) % 30000 for token in tokens] + [102]
            input_ids += [0] * (max_length - len(input_ids))
            attention_mask = [1] * (len(tokens) + 2) + [0] * (max_length - len(tokens) - 2)
            return np.array(input_ids[:max_length], dtype=np.int32), np.array(attention_mask[:max_length], dtype=np.int32)
        
        predictions = []
        inference_times = []
        
        for item in test_data[:100]:  # Test on subset
            input_ids, attention_mask = simple_tokenize(item['text'])
            
            # Set input tensors
            interpreter.set_tensor(input_details[0]['index'], input_ids.reshape(1, -1))
            if len(input_details) > 1:
                interpreter.set_tensor(input_details[1]['index'], attention_mask.reshape(1, -1))
            
            # Run inference
            start_time = time.time()
            try:
                interpreter.invoke()
                inference_times.append(time.time() - start_time)
                
                # Get output
                output = interpreter.get_tensor(output_details[0]['index'])
                pred = np.argmax(output)
                predictions.append(pred)
                
            except Exception as e:
                print(f"Error in prediction: {e}")
                continue
        
        # Calculate metrics
        if len(predictions) > 0:
            avg_inference_time = np.mean(inference_times)
            results = {
                'avg_inference_time': avg_inference_time,
                'min_inference_time': np.min(inference_times),
                'max_inference_time': np.max(inference_times),
                'samples_per_second': 1.0 / avg_inference_time,
                'successful_predictions': len(predictions),
                'total_attempts': len(test_data[:100])
            }
        else:
            results = {'error': 'No successful predictions'}
        
        return results
    
    def benchmark_model(self, model_path: str, platform: str, num_runs: int = 10) -> Dict[str, Any]:
        """Benchmark model performance."""
        print(f"Benchmarking {platform} model...")
        
        # Create dummy test data
        dummy_data = [{'text': f'This is test sentence number {i}.'} for i in range(num_runs)]
        dummy_data_path = '/tmp/dummy_test.json'
        
        with open(dummy_data_path, 'w') as f:
            json.dump(dummy_data, f)
        
        if platform == 'pytorch':
            results = self.evaluate_pytorch_model(model_path, dummy_data_path)
        elif platform == 'ios':
            results = self.evaluate_coreml_model(model_path, dummy_data_path)
        elif platform == 'android':
            results = self.evaluate_tflite_model(model_path, dummy_data_path)
        else:
            results = {'error': f'Unknown platform: {platform}'}
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate BERT models')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model to evaluate')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data')
    parser.add_argument('--platform', type=str, 
                       choices=['pytorch', 'ios', 'android'], 
                       default='pytorch',
                       help='Model platform to evaluate')
    parser.add_argument('--task_type', type=str, 
                       choices=['classification', 'language_modeling'],
                       default='classification',
                       help='Task type')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')
    parser.add_argument('--output', type=str,
                       help='Output file for results')
    
    args = parser.parse_args()
    
    try:
        evaluator = BERTEvaluator()
        
        if args.benchmark:
            results = evaluator.benchmark_model(args.model_path, args.platform)
        else:
            if args.platform == 'pytorch':
                results = evaluator.evaluate_pytorch_model(
                    args.model_path, args.test_data, args.task_type
                )
            elif args.platform == 'ios':
                results = evaluator.evaluate_coreml_model(args.model_path, args.test_data)
            elif args.platform == 'android':
                results = evaluator.evaluate_tflite_model(args.model_path, args.test_data)
        
        # Print results
        print("\\nEvaluation Results:")
        print("=" * 50)
        for key, value in results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            elif isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.4f}")
                    else:
                        print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
        
        # Save results if output specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\\nResults saved to: {args.output}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()"Building custom vocabulary...")
        
        # Initialize vocabulary builder
        builder = VocabularyBuilder(args.config)
        
        # Override config with command line arguments
        builder.vocab_size = args.vocab_size
        builder.min_frequency = args.min_frequency
        
        # Get corpus files
        input_path = Path(args.input_dir)
        if input_path.is_file():
            corpus_paths = [str(input_path)]
        else:
            corpus_paths = [str(input_path)]
        
        # Build vocabulary
        vocab_path = builder.build_from_corpus(
            corpus_paths=corpus_paths,
            output_dir=args.output_dir,
            vocab_name=args.vocab_name
        )
        
        print(f"Vocabulary built successfully: {vocab_path}")
        
        # Analyze domain terms if requested
        if args.analyze_domain:
            texts = builder.load_texts(corpus_paths)
            domain_terms = builder.analyze_domain_terms(
                texts, Path(args.output_dir), top_k=1000
            )
            print(f"Found {len(domain_terms)} domain-specific terms")
        
        print("Vocabulary building completed!")
        
    except Exception as e:
        print(f"Error building vocabulary: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# bert_mobile/scripts/prepare_data.py

#!/usr/bin/env python3

import argparse
import sys
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from tokenizer_utils import BERTDataProcessor

class BERTDataProcessor:
    """Process and prepare data for BERT training."""
    
    def __init__(self, tokenizer_path: str, max_length: int = 512):
        from transformers import BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.max_length = max_length
    
    def process_text_files(self, input_dir: str, output_dir: str):
        """Process text files for language modeling."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_texts = []
        
        # Load text files
        for txt_file in input_path.rglob("*.txt"):
            with open(txt_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f.readlines() if line.strip()]
                all_texts.extend(texts)
        
        print(f"Loaded {len(all_texts)} text samples")
        
        # Process and tokenize
        processed_data = []
        for text in all_texts:
            tokens = self.tokenizer.encode_plus(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            processed_data.append({
                'text': text,
                'input_ids': tokens['input_ids'].squeeze().tolist(),
                'attention_mask': tokens['attention_mask'].squeeze().tolist()
            })
        
        # Split data
        train_size = int(0.8 * len(processed_data))
        val_size = int(0.1 * len(processed_data))
        
        train_data = processed_data[:train_size]
        val_data = processed_data[train_size:train_size + val_size]
        test_data = processed_data[train_size + val_size:]
        
        # Save processed data
        with open(output_path / "train.json", 'w') as f:
            json.dump(train_data, f, indent=2)
        
        with open(output_path / "val.json", 'w') as f:
            json.dump(val_data, f, indent=2)
        
        with open(output_path / "test.json", 'w') as f:
            json.dump(test_data, f, indent=2)
        
        print(f"Processed data saved to {output_path}")
        return str(output_path)
    
    def process_classification_data(self, csv_path: str, output_dir: str, 
                                  text_column: str, label_column: str):
        """Process CSV data for classification."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load CSV
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} samples")
        
        # Process data
        processed_data = []
        label_map = {label: idx for idx, label in enumerate(df[label_column].unique())}
        
        for _, row in df.iterrows():
            text = str(row[text_column])
            label = label_map[row[label_column]]
            
            tokens = self.tokenizer.encode_plus(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            processed_data.append({
                'text': text,
                'label': label,
                'input_ids': tokens['input_ids'].squeeze().tolist(),
                'attention_mask': tokens['attention_mask'].squeeze().tolist()
            })
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        train_data, temp_data = train_test_split(processed_data, test_size=0.3, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        
        # Save processed data
        with open(output_path / "train.json", 'w') as f:
            json.dump(train_data, f, indent=2)
        
        with open(output_path / "val.json", 'w') as f:
            json.dump(val_data, f, indent=2)
        
        with open(output_path / "test.json", 'w') as f:
            json.dump(test_data, f, indent=2)
        
        # Save label mapping
        with open(output_path / "label_map.json", 'w') as f:
            json.dump(label_map, f, indent=2)
        
        print(
