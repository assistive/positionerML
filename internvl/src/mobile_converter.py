# internvl/src/mobile_converter.py

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import json
import yaml

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MobileConverter:
    """Converts InternVL models for mobile deployment."""
    
    def __init__(self, config_path: str = "config/deployment_config.yaml"):
        """
        Initialize mobile converter.
        
        Args:
            config_path: Path to deployment configuration
        """
        self.config = self.load_config(config_path)
        
    def load_config(self, config_path: str) -> Dict:
        """Load deployment configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def convert_to_coreml(self, 
                         model_path: str,
                         output_path: str,
                         input_shape: Tuple[int, ...] = (1, 3, 224, 224),
                         text_input_shape: Tuple[int, ...] = (1, 512)) -> str:
        """
        Convert model to CoreML format for iOS deployment.
        
        Args:
            model_path: Path to trained model
            output_path: Output path for CoreML model
            input_shape: Input image shape (batch, channels, height, width)
            text_input_shape: Input text shape (batch, sequence_length)
            
        Returns:
            Path to converted CoreML model
        """
        if not COREML_AVAILABLE:
            raise ImportError("CoreMLTools not available. Install with: pip install coremltools")
        
        logger.info(f"Converting model to CoreML: {model_path}")
        
        try:
            # Load PyTorch model
            from transformers import AutoModel, AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModel.from_pretrained(
                model_path, 
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
            model.eval()
            
            # Create traced model for CoreML conversion
            traced_model = self.create_traced_model(
                model, input_shape, text_input_shape
            )
            
            # Convert to CoreML
            ios_config = self.config['deployment']['ios']
            
            # Define inputs
            image_input = ct.TensorType(
                name="image",
                shape=input_shape,
                dtype=np.float32
            )
            
            text_input = ct.TensorType(
                name="input_ids", 
                shape=text_input_shape,
                dtype=np.int32
            )
            
            # Convert model
            coreml_model = ct.convert(
                traced_model,
                inputs=[image_input, text_input],
                outputs=[ct.TensorType(name="output")],
                minimum_deployment_target=ct.target.iOS15,
                compute_precision=ct.precision.FLOAT16 if ios_config['precision'] == 'float16' else ct.precision.FLOAT32,
                compute_units=getattr(ct.ComputeUnit, ios_config['compute_units'].upper())
            )
            
            # Set model metadata
            coreml_model.author = ios_config['coreml']['model_author']
            coreml_model.short_description = ios_config['coreml']['model_description']
            coreml_model.version = "1.0"
            
            # Apply optimizations
            if ios_config['optimization']['quantize_weights']:
                coreml_model = ct.models.neural_network.quantization_utils.quantize_weights(
                    coreml_model, nbits=8
                )
            
            # Save model
            output_file = Path(output_path) / f"{ios_config['coreml']['model_name']}.mlmodel"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            coreml_model.save(str(output_file))
            
            logger.info(f"CoreML model saved to: {output_file}")
            
            # Save metadata
            self.save_conversion_metadata(
                output_file.with_suffix('.json'),
                {
                    'platform': 'ios',
                    'format': 'coreml',
                    'input_shape': input_shape,
                    'text_input_shape': text_input_shape,
                    'model_path': model_path,
                    'config': ios_config
                }
            )
            
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error converting to CoreML: {e}")
            raise
    
    def convert_to_tflite(self,
                         model_path: str,
                         output_path: str,
                         input_shape: Tuple[int, ...] = (1, 3, 224, 224),
                         text_input_shape: Tuple[int, ...] = (1, 512)) -> str:
        """
        Convert model to TensorFlow Lite format for Android deployment.
        
        Args:
            model_path: Path to trained model
            output_path: Output path for TFLite model
            input_shape: Input image shape
            text_input_shape: Input text shape
            
        Returns:
            Path to converted TFLite model
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available. Install with: pip install tensorflow")
        
        logger.info(f"Converting model to TensorFlow Lite: {model_path}")
        
        try:
            # First convert to ONNX, then to TensorFlow, then to TFLite
            onnx_path = self.convert_to_onnx(model_path, input_shape, text_input_shape)
            tf_model_path = self.onnx_to_tensorflow(onnx_path)
            tflite_path = self.tensorflow_to_tflite(tf_model_path, output_path)
            
            return tflite_path
            
        except Exception as e:
            logger.error(f"Error converting to TFLite: {e}")
            raise
    
    def convert_to_onnx(self,
                       model_path: str,
                       input_shape: Tuple[int, ...] = (1, 3, 224, 224),
                       text_input_shape: Tuple[int, ...] = (1, 512)) -> str:
        """Convert PyTorch model to ONNX format."""
        logger.info("Converting to ONNX...")
        
        from transformers import AutoModel, AutoTokenizer
        
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32
        )
        model.eval()
        
        # Create dummy inputs
        dummy_image = torch.randn(input_shape)
        dummy_text = torch.randint(0, tokenizer.vocab_size, text_input_shape)
        
        # ONNX export path
        onnx_path = Path(model_path).parent / "model.onnx"
        
        # Export to ONNX
        torch.onnx.export(
            model,
            (dummy_image, dummy_text),
            str(onnx_path),
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['image', 'input_ids'],
            output_names=['output'],
            dynamic_axes={
                'image': {0: 'batch_size'},
                'input_ids': {0: 'batch_size'},
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
    
    def tensorflow_to_tflite(self, tf_model_path: str, output_path: str) -> str:
        """Convert TensorFlow model to TFLite."""
        android_config = self.config['deployment']['android']
        
        # Load TensorFlow model
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
        
        # Apply optimizations
        if android_config['optimization']['quantize']:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Use representative dataset for quantization
            def representative_dataset():
                for _ in range(android_config['tflite']['representative_dataset_size']):
                    # Create dummy data
                    image = np.random.random((1, 3, 224, 224)).astype(np.float32)
                    text = np.random.randint(0, 1000, (1, 512)).astype(np.int32)
                    yield [image, text]
            
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.TFLITE_BUILTINS
            ]
        
        # Convert
        tflite_model = converter.convert()
        
        # Save TFLite model
        tflite_path = Path(output_path) / "model.tflite"
        tflite_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        logger.info(f"TFLite model saved to: {tflite_path}")
        
        # Save metadata
        self.save_conversion_metadata(
            tflite_path.with_suffix('.json'),
            {
                'platform': 'android',
                'format': 'tflite',
                'input_shape': (1, 3, 224, 224),
                'text_input_shape': (1, 512),
                'model_path': tf_model_path,
                'config': android_config
            }
        )
        
        return str(tflite_path)
    
    def create_traced_model(self, 
                           model: torch.nn.Module,
                           input_shape: Tuple[int, ...],
                           text_input_shape: Tuple[int, ...]) -> torch.jit.ScriptModule:
        """Create a traced PyTorch model for conversion."""
        
        # Create wrapper that handles the specific input format
        class ModelWrapper(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.model = base_model
            
            def forward(self, image, input_ids):
                # Adapt inputs to match model's expected format
                return self.model(images=image, input_ids=input_ids)
        
        wrapped_model = ModelWrapper(model)
        
        # Create dummy inputs
        dummy_image = torch.randn(input_shape)
        dummy_text = torch.randint(0, 1000, text_input_shape)
        
        # Trace the model
        traced_model = torch.jit.trace(
            wrapped_model, 
            (dummy_image, dummy_text)
        )
        
        return traced_model
    
    def optimize_for_mobile(self, 
                           model_path: str,
                           optimization_config: Dict) -> torch.jit.ScriptModule:
        """Apply mobile-specific optimizations to the model."""
        
        # Load model
        model = torch.jit.load(model_path)
        
        # Apply optimizations
        if optimization_config.get('quantization', {}).get('enabled', False):
            # Dynamic quantization
            model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
        
        # Optimize for mobile
        optimized_model = torch.utils.mobile_optimizer.optimize_for_mobile(model)
        
        return optimized_model
    
    def validate_conversion(self, 
                           original_model_path: str,
                           converted_model_path: str,
                           platform: str) -> Dict[str, float]:
        """Validate the converted model against the original."""
        logger.info(f"Validating {platform} conversion...")
        
        try:
            # Create test inputs
            test_image = np.random.random((1, 3, 224, 224)).astype(np.float32)
            test_text = np.random.randint(0, 1000, (1, 512)).astype(np.int32)
            
            # Get original model output
            from transformers import AutoModel, AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(original_model_path, trust_remote_code=True)
            original_model = AutoModel.from_pretrained(
                original_model_path,
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
            original_model.eval()
            
            with torch.no_grad():
                original_output = original_model(
                    images=torch.from_numpy(test_image),
                    input_ids=torch.from_numpy(test_text)
                )
            
            # Get converted model output
            if platform == 'ios' and COREML_AVAILABLE:
                converted_model = ct.models.model.MLModel(converted_model_path)
                converted_output = converted_model.predict({
                    'image': test_image,
                    'input_ids': test_text
                })
                
            elif platform == 'android' and TF_AVAILABLE:
                interpreter = tf.lite.Interpreter(model_path=converted_model_path)
                interpreter.allocate_tensors()
                
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                interpreter.set_tensor(input_details[0]['index'], test_image)
                interpreter.set_tensor(input_details[1]['index'], test_text)
                
                interpreter.invoke()
                
                converted_output = interpreter.get_tensor(output_details[0]['index'])
            
            # Calculate metrics
            if isinstance(original_output, torch.Tensor):
                original_np = original_output.numpy()
            else:
                original_np = original_output.last_hidden_state.numpy()
            
            if isinstance(converted_output, dict):
                converted_np = list(converted_output.values())[0]
            else:
                converted_np = converted_output
            
            # Ensure same shape for comparison
            if original_np.shape != converted_np.shape:
                logger.warning(f"Shape mismatch: original {original_np.shape} vs converted {converted_np.shape}")
                # Try to match shapes
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

