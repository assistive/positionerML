"""
DINOv2 Mobile Converter
Converts DINOv2 models to mobile-optimized formats (CoreML for iOS, TFLite for Android)
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union
import logging

# Optional imports for mobile conversion
try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


class DINOv2MobileConverter:
    """Convert DINOv2 models for mobile deployment."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        
    def load_pytorch_model(self, model_path: str) -> None:
        """Load PyTorch DINOv2 model."""
        if model_path.endswith('.pth'):
            self.model = torch.load(model_path, map_location='cpu')
        else:
            # Load from torch.hub
            self.model = torch.hub.load('facebookresearch/dinov2', model_path)
        
        self.model.eval()
        
    def convert_to_coreml(self, output_path: str) -> str:
        """Convert model to CoreML format for iOS."""
        if not COREML_AVAILABLE:
            raise ImportError("CoreML Tools not available. Install with: pip install coremltools")
            
        logger.info("Converting to CoreML format...")
        
        # Create example input
        example_input = torch.randn(1, 3, 224, 224)
        
        # Trace the model
        traced_model = torch.jit.trace(self.model, example_input)
        
        # Convert to CoreML
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.ImageType(
                name="image",
                shape=example_input.shape,
                bias=[-0.485/-0.229, -0.456/-0.224, -0.406/-0.225],
                scale=[1/(0.229*255.0), 1/(0.224*255.0), 1/(0.225*255.0)]
            )],
            outputs=[ct.TensorType(name="features")],
            compute_units=ct.ComputeUnit.ALL
        )
        
        # Add metadata
        coreml_model.short_description = "DINOv2 Vision Transformer for mobile"
        coreml_model.author = "Meta Research (DINOv2), Converted for mobile"
        coreml_model.license = "Apache 2.0"
        
        # Save model
        output_file = f"{output_path}/dinov2_mobile.mlpackage"
        coreml_model.save(output_file)
        
        logger.info(f"CoreML model saved to: {output_file}")
        return output_file
        
    def convert_to_tflite(self, output_path: str) -> str:
        """Convert model to TensorFlow Lite format for Android."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available. Install with: pip install tensorflow")
            
        logger.info("Converting to TensorFlow Lite format...")
        
        # Convert PyTorch to ONNX first, then to TensorFlow
        import torch.onnx
        import onnx
        import onnx_tf
        
        # Export to ONNX
        example_input = torch.randn(1, 3, 224, 224)
        onnx_path = f"{output_path}/dinov2_temp.onnx"
        
        torch.onnx.export(
            self.model,
            example_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'},
                         'output': {0: 'batch_size'}}
        )
        
        # Convert ONNX to TensorFlow
        onnx_model = onnx.load(onnx_path)
        tf_rep = onnx_tf.backend.prepare(onnx_model)
        tf_model_path = f"{output_path}/dinov2_tf"
        tf_rep.export_graph(tf_model_path)
        
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
        
        # Optimization settings
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        # Post-training quantization
        if self.config.get('quantization', {}).get('enabled', False):
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = self._get_representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
        
        tflite_model = converter.convert()
        
        # Save TFLite model
        output_file = f"{output_path}/dinov2_mobile.tflite"
        with open(output_file, 'wb') as f:
            f.write(tflite_model)
            
        # Cleanup temporary files
        Path(onnx_path).unlink()
        import shutil
        shutil.rmtree(tf_model_path)
        
        logger.info(f"TensorFlow Lite model saved to: {output_file}")
        return output_file
    
    def _get_representative_dataset(self):
        """Generate representative dataset for quantization."""
        for _ in range(100):
            yield [np.random.random((1, 3, 224, 224)).astype(np.float32)]
    
    def validate_conversion(self, original_output: torch.Tensor, 
                          converted_output: np.ndarray) -> Dict:
        """Validate model conversion accuracy."""
        # Convert tensors to same format
        original_np = original_output.detach().cpu().numpy()
        
        # Calculate metrics
        mse = np.mean((original_np - converted_output) ** 2)
        cosine_sim = np.dot(original_np.flatten(), converted_output.flatten()) / \
                    (np.linalg.norm(original_np) * np.linalg.norm(converted_output))
        
        return {
            "mse": float(mse),
            "cosine_similarity": float(cosine_sim),
            "max_diff": float(np.max(np.abs(original_np - converted_output))),
            "status": "success" if cosine_sim > 0.95 else "warning"
        }
