"""
MobileCLIP Mobile Converter
Converts MobileCLIP models to mobile-optimized formats (CoreML, TensorFlow Lite)
"""
import torch
import torch.nn as nn
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np

class MobileCLIPMobileConverter:
    """Convert MobileCLIP models for mobile deployment."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.mobile_model = None
        
    def load_pytorch_model(self, model_name: str, model_path: Optional[str] = None):
        """Load PyTorch MobileCLIP model."""
        try:
            import mobileclip
            from mobileclip.modules.common.mobileone import reparameterize_model
            
            # Load model
            self.model, _, _ = mobileclip.create_model_and_transforms(
                model_name, pretrained=model_path
            )
            
            # Reparameterize for inference
            self.model.eval()
            self.mobile_model = reparameterize_model(self.model)
            self.mobile_model.eval()
            
            logging.info(f"Loaded and reparameterized {model_name}")
            
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise
    
    def convert_to_coreml(self, output_dir: str, model_name: str = "mobileclip") -> str:
        """Convert model to CoreML format for iOS."""
        try:
            import coremltools as ct
        except ImportError:
            raise ImportError("CoreML Tools not available. Install with: pip install coremltools")
        
        if self.mobile_model is None:
            raise RuntimeError("No model loaded")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create example inputs
        example_image = torch.randn(1, 3, 224, 224)
        example_text = torch.randint(0, 1000, (1, 77))  # Text tokens
        
        # Trace the model
        with torch.no_grad():
            # Split into image and text encoders for separate conversion
            traced_image_model = torch.jit.trace(
                self.mobile_model.visual, 
                example_image
            )
            traced_text_model = torch.jit.trace(
                self.mobile_model.text, 
                example_text
            )
        
        # Convert image encoder
        image_model_path = output_path / f"{model_name}_image.mlpackage"
        image_coreml_model = ct.convert(
            traced_image_model,
            inputs=[ct.ImageType(name="image", shape=example_image.shape)],
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.iOS15
        )
        image_coreml_model.save(str(image_model_path))
        
        # Convert text encoder  
        text_model_path = output_path / f"{model_name}_text.mlpackage"
        text_coreml_model = ct.convert(
            traced_text_model,
            inputs=[ct.TensorType(name="text_tokens", shape=example_text.shape)],
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.iOS15
        )
        text_coreml_model.save(str(text_model_path))
        
        logging.info(f"CoreML models saved to {output_path}")
        return str(output_path)
    
    def convert_to_tflite(self, output_dir: str, model_name: str = "mobileclip") -> str:
        """Convert model to TensorFlow Lite format for Android."""
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow not available. Install with: pip install tensorflow")
        
        if self.mobile_model is None:
            raise RuntimeError("No model loaded")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert through ONNX first
        onnx_path = self._convert_to_onnx(output_path, model_name)
        
        # Convert ONNX to TensorFlow Lite
        tflite_path = self._onnx_to_tflite(onnx_path, output_path, model_name)
        
        logging.info(f"TensorFlow Lite model saved to {tflite_path}")
        return str(tflite_path)
    
    def _convert_to_onnx(self, output_dir: Path, model_name: str) -> str:
        """Convert PyTorch model to ONNX."""
        try:
            import torch.onnx
        except ImportError:
            raise ImportError("ONNX not available. Install with: pip install onnx")
        
        # Example inputs
        example_image = torch.randn(1, 3, 224, 224)
        example_text = torch.randint(0, 1000, (1, 77))
        
        # Export image encoder
        image_onnx_path = output_dir / f"{model_name}_image.onnx"
        torch.onnx.export(
            self.mobile_model.visual,
            example_image,
            str(image_onnx_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['image'],
            output_names=['image_features']
        )
        
        # Export text encoder
        text_onnx_path = output_dir / f"{model_name}_text.onnx"
        torch.onnx.export(
            self.mobile_model.text,
            example_text,
            str(text_onnx_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['text_tokens'],
            output_names=['text_features']
        )
        
        return str(image_onnx_path)
    
    def _onnx_to_tflite(self, onnx_path: str, output_dir: Path, model_name: str) -> str:
        """Convert ONNX model to TensorFlow Lite."""
        try:
            import onnx
            import tensorflow as tf
            from onnx_tf.backend import prepare
        except ImportError:
            raise ImportError("ONNX-TF not available. Install with: pip install onnx-tf")
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Convert to TensorFlow
        tf_rep = prepare(onnx_model)
        
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_concrete_functions(tf_rep.signatures)
        
        # Optimization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        
        # Save model
        tflite_path = output_dir / f"{model_name}.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        return str(tflite_path)
    
    def validate_conversion(self, original_output: torch.Tensor, converted_output: np.ndarray, tolerance: float = 1e-3) -> bool:
        """Validate model conversion accuracy."""
        original_np = original_output.detach().cpu().numpy()
        
        # Compare outputs
        diff = np.abs(original_np - converted_output)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        logging.info(f"Conversion validation - Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
        
        return max_diff < tolerance
