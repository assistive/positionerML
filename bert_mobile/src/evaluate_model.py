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
            try:
                input_ids_array = ct.models.utils.MLMultiArray(
                    np.array(input_ids, dtype=np.int32).reshape(1, -1)
                )
                attention_mask_array = ct.models.utils.MLMultiArray(
                    np.array(attention_mask, dtype=np.int32).reshape(1, -1)
                )
                
                # Run inference
                start_time = time.time()
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
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed evaluation metrics')
    
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
        print("\nEvaluation Results:")
        print("=" * 50)
        for key, value in results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            elif isinstance(value, dict) and args.detailed:
                print(f"{key}:")
                for k, v in value.items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.4f}")
                    else:
                        print(f"  {k}: {v}")
            elif not isinstance(value, dict):
                print(f"{key}: {value}")
        
        # Print summary
        if 'accuracy' in results:
            accuracy = results['accuracy']
            if accuracy > 0.9:
                print(f"\n✅ Excellent accuracy: {accuracy:.2%}")
            elif accuracy > 0.8:
                print(f"\n✅ Good accuracy: {accuracy:.2%}")
            elif accuracy > 0.7:
                print(f"\n⚠️  Acceptable accuracy: {accuracy:.2%}")
            else:
                print(f"\n❌ Low accuracy: {accuracy:.2%}")
        
        if 'avg_inference_time' in results:
            inference_time = results['avg_inference_time']
            if inference_time < 0.1:
                print(f"⚡ Fast inference: {inference_time*1000:.1f}ms")
            elif inference_time < 0.5:
                print(f"✅ Good inference speed: {inference_time*1000:.1f}ms")
            else:
                print(f"⚠️  Slow inference: {inference_time*1000:.1f}ms")
        
        # Save results if output specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to: {args.output}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


