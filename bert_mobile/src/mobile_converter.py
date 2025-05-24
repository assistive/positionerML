# bert_mobile/src/mobile_converter.py

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import yaml
import tempfile
import shutil

# Mobile deployment imports
try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False
    logging.warning("CoreMLTools not available - iOS deployment disabled")

try:
    import tensorflow as tf
    import onnx
    import onnxruntime as ort
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow/ONNX not available - Android deployment disabled")

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BERTMobileConverter:
    """Convert BERT models for mobile deployment."""
    
    def __init__(self, config_path: str = "config/mobile_config.yaml"):
        """
        Initialize mobile converter.
        
        Args:
            config_path: Path to mobile deployment configuration
        """
        self.config = self.load_config(config_path)
        
    def load_config(self, config_path: str) -> Dict:
        """Load mobile deployment configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def convert_to_coreml(self, 
                         model_path: str,
                         output_path: str,
                         sequence_length: int = 128,
                         task_type: str = "classification") -> str:
        """
        Convert BERT model to CoreML format for iOS deployment.
        
        Args:
            model_path: Path to trained BERT model
            output_path: Output path for CoreML model
            sequence_length: Maximum sequence length
            task_type: Type of task ("classification" or "feature_extraction")
            
        Returns:
            Path to converted CoreML model
        """
        if not COREML_AVAILABLE:
            raise ImportError("CoreMLTools not available. Install with: pip install coremltools")
        
        logger.info(f"Converting BERT model to CoreML: {model_path}")
        
        try:
            # Load PyTorch model
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            if task_type == "classification":
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
            else:
                model = AutoModel.from_pretrained(model_path)
            
            model.eval()
            
            # Create traced model for CoreML conversion
            traced_model = self.create_traced_bert_model(model, sequence_length)
            
            # Convert to CoreML
            ios_config = self.config['mobile']['ios']
            
            # Define inputs
            input_ids = ct.TensorType(
                name="input_ids",
                shape=(1, sequence_length),
                dtype=np.int32
            )
            
            attention_mask = ct.TensorType(
                name="attention_mask", 
                shape=(1, sequence_length),
                dtype=np.int32
            )
            
            # Convert model
            coreml_model = ct.convert(
                traced_model,
                inputs=[input_ids, attention_mask],
                outputs=[ct.TensorType(name="output")],
                minimum_deployment_target=getattr(ct.target, ios_config['deployment_target'].replace('.', '')),
                compute_precision=getattr(ct.precision, ios_config['precision'].upper()),
                compute_units=getattr(ct.ComputeUnit, ios_config['compute_units'].upper())
            )
            
            # Set model metadata
            coreml_config = ios_config['coreml']
            coreml_model.author = coreml_config['model_author']
            coreml_model.short_description = coreml_config['model_description']
            coreml_model.version = "1.0"
            
            # Apply optimizations
            if ios_config['optimization']['quantize_weights']:
                coreml_model = ct.models.neural_network.quantization_utils.quantize_weights(
                    coreml_model, nbits=8
                )
            
            if ios_config['optimization']['compress_weights']:
                coreml_model = ct.models.neural_network.quantization_utils.quantize_weights(
                    coreml_model, nbits=16
                )
            
            # Save model
            output_file = Path(output_path) / f"{coreml_config['model_name']}.mlmodel"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            coreml_model.save(str(output_file))
            
            logger.info(f"CoreML model saved to: {output_file}")
            
            # Save metadata and tokenizer info
            self.save_conversion_metadata(
                output_file.with_suffix('.json'),
                {
                    'platform': 'ios',
                    'format': 'coreml',
                    'sequence_length': sequence_length,
                    'task_type': task_type,
                    'model_path': model_path,
                    'config': ios_config,
                    'vocab_size': len(tokenizer.vocab),
                    'tokenizer_info': self.extract_tokenizer_info(tokenizer)
                }
            )
            
            # Save tokenizer for iOS
            self.save_tokenizer_for_ios(tokenizer, output_file.parent)
            
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error converting to CoreML: {e}")
            raise
    
    def convert_to_tflite(self,
                         model_path: str,
                         output_path: str,
                         sequence_length: int = 128,
                         task_type: str = "classification") -> str:
        """
        Convert BERT model to TensorFlow Lite format for Android deployment.
        
        Args:
            model_path: Path to trained BERT model
            output_path: Output path for TFLite model
            sequence_length: Maximum sequence length
            task_type: Type of task
            
        Returns:
            Path to converted TFLite model
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available. Install with: pip install tensorflow")
        
        logger.info(f"Converting BERT model to TensorFlow Lite: {model_path}")
        
        try:
            # Convert via ONNX to TensorFlow to TFLite pipeline
            onnx_path = self.convert_to_onnx(model_path, sequence_length, task_type)
            tf_model_path = self.onnx_to_tensorflow(onnx_path)
            tflite_path = self.tensorflow_to_tflite(tf_model_path, output_path, model_path)
            
            return tflite_path
            
        except Exception as e:
            logger.error(f"Error converting to TFLite: {e}")
            raise
    
    def convert_to_onnx(self,
                       model_path: str,
                       sequence_length: int = 128,
                       task_type: str = "classification") -> str:
        """Convert PyTorch BERT model to ONNX format."""
        logger.info("Converting to ONNX...")
        
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if task_type == "classification":
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            model = AutoModel.from_pretrained(model_path)
        
        model.eval()
        
        # Create dummy inputs
        dummy_input_ids = torch.ones((1, sequence_length), dtype=torch.long)
        dummy_attention_mask = torch.ones((1, sequence_length), dtype=torch.long)
        
        # ONNX export path
        onnx_path = Path(model_path).parent / "model.onnx"
        
        # Export to ONNX
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask),
            str(onnx_path),
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask'],
            output_names=['output'],
            dynamic_axes={
                'input_ids': {0: 'batch_size'},
                'attention_mask': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        logger.info(f"ONNX model saved to: {onnx_path}")
        return str(onnx_path)
    
    def onnx_to_tensorflow(self, onnx_path: str) -> str:
        """Convert ONNX model to TensorFlow."""
        try:
            import onnx_tf
            
            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            
            # Convert to TensorFlow
            tf_rep = onnx_tf.backend.prepare(onnx_model)
            
            # Save TensorFlow model
            tf_model_path = Path(onnx_path).parent / "tf_model"
            tf_rep.export_graph(str(tf_model_path))
            
            logger.info(f"TensorFlow model saved to: {tf_model_path}")
            return str(tf_model_path)
            
        except ImportError:
            logger.error("onnx-tf not available. Install with: pip install onnx-tf")
            raise
    
    def tensorflow_to_tflite(self, tf_model_path: str, output_path: str, original_model_path: str) -> str:
        """Convert TensorFlow model to TFLite."""
        android_config = self.config['mobile']['android']
        
        # Load TensorFlow model
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
        
        # Apply optimizations
        if android_config['optimization']['quantization'] != "none":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            if android_config['optimization']['quantization'] == "int8":
                # Use representative dataset for quantization
                def representative_dataset():
                    for _ in range(android_config['optimization']['representative_dataset_size']):
                        # Create dummy data
                        input_ids = np.random.randint(0, 1000, (1, 128)).astype(np.int32)
                        attention_mask = np.ones((1, 128)).astype(np.int32)
                        yield [input_ids, attention_mask]
                
                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            
            elif android_config['optimization']['quantization'] == "float16":
                converter.target_spec.supported_types = [tf.float16]
        
        # Set supported ops
        tflite_config = android_config['tflite']
        supported_ops = []
        for op_set in tflite_config['supported_ops']:
            if hasattr(tf.lite.OpsSet, op_set):
                supported_ops.append(getattr(tf.lite.OpsSet, op_set))
        
        if supported_ops:
            converter.target_spec.supported_ops = supported_ops
        
        converter.allow_custom_ops = tflite_config.get('allow_custom_ops', False)
        
        # Convert
        tflite_model = converter.convert()
        
        # Save TFLite model
        tflite_path = Path(output_path) / "bert_mobile.tflite"
        tflite_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        logger.info(f"TFLite model saved to: {tflite_path}")
        
        # Save metadata and tokenizer info
        tokenizer = AutoTokenizer.from_pretrained(original_model_path)
        self.save_conversion_metadata(
            tflite_path.with_suffix('.json'),
            {
                'platform': 'android',
                'format': 'tflite',
                'sequence_length': 128,
                'model_path': original_model_path,
                'config': android_config,
                'vocab_size': len(tokenizer.vocab),
                'tokenizer_info': self.extract_tokenizer_info(tokenizer)
            }
        )
        
        # Save tokenizer for Android
        self.save_tokenizer_for_android(tokenizer, tflite_path.parent)
        
        return str(tflite_path)
    
    def create_traced_bert_model(self, 
                                model: torch.nn.Module,
                                sequence_length: int) -> torch.jit.ScriptModule:
        """Create a traced PyTorch model for conversion."""
        
        # Create wrapper that handles the specific input format
        class BERTWrapper(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.model = base_model
            
            def forward(self, input_ids, attention_mask):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                # Return logits for classification or last hidden state for feature extraction
                if hasattr(outputs, 'logits'):
                    return outputs.logits
                else:
                    return outputs.last_hidden_state.mean(dim=1)  # Pool sequence dimension
        
        wrapped_model = BERTWrapper(model)
        
        # Create dummy inputs
        dummy_input_ids = torch.ones((1, sequence_length), dtype=torch.long)
        dummy_attention_mask = torch.ones((1, sequence_length), dtype=torch.long)
        
        # Trace the model
        traced_model = torch.jit.trace(
            wrapped_model, 
            (dummy_input_ids, dummy_attention_mask)
        )
        
        return traced_model
    
    def extract_tokenizer_info(self, tokenizer) -> Dict[str, Any]:
        """Extract tokenizer information for mobile deployment."""
        return {
            'vocab_size': len(tokenizer.vocab),
            'max_length': tokenizer.model_max_length,
            'pad_token': tokenizer.pad_token,
            'cls_token': tokenizer.cls_token,
            'sep_token': tokenizer.sep_token,
            'unk_token': tokenizer.unk_token,
            'mask_token': getattr(tokenizer, 'mask_token', None),
            'pad_token_id': tokenizer.pad_token_id,
            'cls_token_id': tokenizer.cls_token_id,
            'sep_token_id': tokenizer.sep_token_id,
            'unk_token_id': tokenizer.unk_token_id,
            'mask_token_id': getattr(tokenizer, 'mask_token_id', None),
            'do_lower_case': getattr(tokenizer, 'do_lower_case', True)
        }
    
    def save_tokenizer_for_ios(self, tokenizer, output_dir: Path):
        """Save tokenizer files optimized for iOS."""
        ios_tokenizer_dir = output_dir / "tokenizer"
        ios_tokenizer_dir.mkdir(exist_ok=True)
        
        # Save vocabulary as simple text file
        vocab = tokenizer.get_vocab()
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        
        vocab_path = ios_tokenizer_dir / "vocab.txt"
        with open(vocab_path, 'w', encoding='utf-8') as f:
            for token, _ in sorted_vocab:
                f.write(f"{token}\n")
        
        # Save tokenizer configuration
        config = {
            'vocab_size': len(vocab),
            'special_tokens': self.extract_tokenizer_info(tokenizer),
            'do_lower_case': getattr(tokenizer, 'do_lower_case', True)
        }
        
        config_path = ios_tokenizer_dir / "tokenizer_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"iOS tokenizer saved to {ios_tokenizer_dir}")
    
    def save_tokenizer_for_android(self, tokenizer, output_dir: Path):
        """Save tokenizer files optimized for Android."""
        android_tokenizer_dir = output_dir / "tokenizer"
        android_tokenizer_dir.mkdir(exist_ok=True)
        
        # Save vocabulary as JSON for easier parsing in Android
        vocab = tokenizer.get_vocab()
        
        vocab_path = android_tokenizer_dir / "vocab.json"
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        
        # Save reverse vocabulary (id to token)
        reverse_vocab = {v: k for k, v in vocab.items()}
        reverse_vocab_path = android_tokenizer_dir / "reverse_vocab.json"
        with open(reverse_vocab_path, 'w', encoding='utf-8') as f:
            json.dump(reverse_vocab, f, ensure_ascii=False, indent=2)
        
        # Save tokenizer configuration
        config = {
            'vocab_size': len(vocab),
            'special_tokens': self.extract_tokenizer_info(tokenizer),
            'do_lower_case': getattr(tokenizer, 'do_lower_case', True)
        }
        
        config_path = android_tokenizer_dir / "tokenizer_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Android tokenizer saved to {android_tokenizer_dir}")
    
    def optimize_for_mobile(self, 
                           model_path: str,
                           optimization_config: Dict) -> str:
        """Apply mobile-specific optimizations to the model."""
        logger.info("Applying mobile optimizations...")
        
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        
        optimized_model_path = Path(model_path).parent / "optimized_model"
        optimized_model_path.mkdir(exist_ok=True)
        
        # Apply quantization if enabled
        if optimization_config.get('quantization', {}).get('enabled', False):
            model = self.apply_quantization(model, optimization_config['quantization'])
        
        # Apply pruning if enabled
        if optimization_config.get('pruning', {}).get('enabled', False):
            model = self.apply_pruning(model, optimization_config['pruning'])
        
        # Save optimized model
        model.save_pretrained(str(optimized_model_path))
        tokenizer.save_pretrained(str(optimized_model_path))
        
        return str(optimized_model_path)
    
    def apply_quantization(self, model: torch.nn.Module, quantization_config: Dict) -> torch.nn.Module:
        """Apply quantization to the model."""
        quantization_method = quantization_config.get('method', 'dynamic')
        
        if quantization_method == 'dynamic':
            # Dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
        elif quantization_method == 'static':
            # Static quantization (requires calibration data)
            logger.warning("Static quantization not implemented in this example")
            quantized_model = model
        else:
            logger.warning(f"Unknown quantization method: {quantization_method}")
            quantized_model = model
        
        logger.info(f"Applied {quantization_method} quantization")
        return quantized_model
    
    def apply_pruning(self, model: torch.nn.Module, pruning_config: Dict) -> torch.nn.Module:
        """Apply pruning to the model."""
        try:
            import torch.nn.utils.prune as prune
            
            sparsity = pruning_config.get('sparsity', 0.5)
            
            # Apply unstructured pruning to linear layers
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=sparsity)
                    prune.remove(module, 'weight')
            
            logger.info(f"Applied pruning with sparsity {sparsity}")
            
        except ImportError:
            logger.warning("Pruning not available in this PyTorch version")
        
        return model
    
    def validate_conversion(self, 
                           original_model_path: str,
                           converted_model_path: str,
                           platform: str) -> Dict[str, Any]:
        """Validate the converted model against the original."""
        logger.info(f"Validating {platform} conversion...")
        
        try:
            # Create test inputs
            test_input_ids = np.random.randint(0, 1000, (1, 128)).astype(np.int32)
            test_attention_mask = np.ones((1, 128)).astype(np.int32)
            
            # Get original model output
            tokenizer = AutoTokenizer.from_pretrained(original_model_path)
            original_model = AutoModel.from_pretrained(original_model_path)
            original_model.eval()
            
            with torch.no_grad():
                original_output = original_model(
                    input_ids=torch.from_numpy(test_input_ids),
                    attention_mask=torch.from_numpy(test_attention_mask)
                )
            
            # Get converted model output
            if platform == 'ios' and COREML_AVAILABLE:
                converted_model = ct.models.model.MLModel(converted_model_path)
                converted_output = converted_model.predict({
                    'input_ids': test_input_ids,
                    'attention_mask': test_attention_mask
                })
                
            elif platform == 'android' and TF_AVAILABLE:
                interpreter = tf.lite.Interpreter(model_path=converted_model_path)
                interpreter.allocate_tensors()
                
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                interpreter.set_tensor(input_details[0]['index'], test_input_ids)
                interpreter.set_tensor(input_details[1]['index'], test_attention_mask)
                
                interpreter.invoke()
                
                converted_output = interpreter.get_tensor(output_details[0]['index'])
            
            # Calculate metrics
            if hasattr(original_output, 'last_hidden_state'):
                original_np = original_output.last_hidden_state.numpy()
            else:
                original_np = original_output.numpy()
            
            if isinstance(converted_output, dict):
                converted_np = list(converted_output.values())[0]
            else:
                converted_np = converted_output
            
            # Ensure same shape for comparison
            if original_np.shape != converted_np.shape:
                logger.warning(f"Shape mismatch: original {original_np.shape} vs converted {converted_np.shape}")
                # Try to match shapes by flattening
                min_size = min(original_np.size, converted_np.size)
                original_np = original_np.flatten()[:min_size]
                converted_np = converted_np.flatten()[:min_size]
            
            # Calculate error metrics
            mse = np.mean((original_np - converted_np) ** 2)
            mae = np.mean(np.abs(original_np - converted_np))
            max_error = np.max(np.abs(original_np - converted_np))
            
            metrics = {
                'mse': float(mse),
                'mae': float(mae),
                'max_error': float(max_error),
                'conversion_successful': True
            }
            
            logger.info(f"Validation metrics: MSE={mse:.6f}, MAE={mae:.6f}, Max Error={max_error:.6f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                'mse': float('inf'),
                'mae': float('inf'),
                'max_error': float('inf'),
                'conversion_successful': False,
                'error': str(e)
            }
    
    def save_conversion_metadata(self, output_path: Path, metadata: Dict):
        """Save conversion metadata."""
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Conversion metadata saved to: {output_path}")
    
    def get_model_size(self, model_path: str) -> Dict[str, float]:
        """Get model file size information."""
        path = Path(model_path)
        
        if path.is_file():
            size_bytes = path.stat().st_size
        else:
            # Directory with multiple files
            size_bytes = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        
        return {
            'size_bytes': size_bytes,
            'size_mb': size_bytes / (1024 * 1024),
            'size_gb': size_bytes / (1024 * 1024 * 1024)
        }
    
    def benchmark_model(self, model_path: str, platform: str, num_runs: int = 10) -> Dict[str, float]:
        """Benchmark converted model performance."""
        logger.info(f"Benchmarking {platform} model...")
        
        # Create test input
        test_input_ids = np.random.randint(0, 1000, (1, 128)).astype(np.int32)
        test_attention_mask = np.ones((1, 128)).astype(np.int32)
        
        times = []
        
        if platform == 'ios' and COREML_AVAILABLE:
            model = ct.models.model.MLModel(model_path)
            
            # Warmup
            for _ in range(2):
                _ = model.predict({
                    'input_ids': test_input_ids,
                    'attention_mask': test_attention_mask
                })
            
            # Benchmark
            import time
            for _ in range(num_runs):
                start_time = time.time()
                _ = model.predict({
                    'input_ids': test_input_ids,
                    'attention_mask': test_attention_mask
                })
                end_time = time.time()
                times.append(end_time - start_time)
        
        elif platform == 'android' and TF_AVAILABLE:
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Warmup
            for _ in range(2):
                interpreter.set_tensor(input_details[0]['index'], test_input_ids)
                interpreter.set_tensor(input_details[1]['index'], test_attention_mask)
                interpreter.invoke()
            
            # Benchmark
            import time
            for _ in range(num_runs):
                start_time = time.time()
                interpreter.set_tensor(input_details[0]['index'], test_input_ids)
                interpreter.set_tensor(input_details[1]['index'], test_attention_mask)
                interpreter.invoke()
                end_time = time.time()
                times.append(end_time - start_time)
        
        if times:
            return {
                'avg_inference_time_ms': np.mean(times) * 1000,
                'std_inference_time_ms': np.std(times) * 1000,
                'min_inference_time_ms': np.min(times) * 1000,
                'max_inference_time_ms': np.max(times) * 1000,
                'fps': 1.0 / np.mean(times)
            }
        else:
            return {}
