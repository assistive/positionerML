#!/usr/bin/env python3
"""
Fixed MobileCLIP Mobile Converter
Handles missing dependencies gracefully and provides clear error messages
"""
import torch
import torch.nn as nn
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np

# Check for MobileCLIP availability
try:
    import mobileclip
    from mobileclip.modules.common.mobileone import reparameterize_model
    MOBILECLIP_AVAILABLE = True
except ImportError as e:
    MOBILECLIP_AVAILABLE = False
    MOBILECLIP_IMPORT_ERROR = str(e)

class MobileCLIPMobileConverter:
    """Convert MobileCLIP models for mobile deployment with better error handling."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.mobile_model = None
        self._check_dependencies()
        
    def _check_dependencies(self):
        """Check if all required dependencies are available."""
        missing_deps = []
        
        if not MOBILECLIP_AVAILABLE:
            missing_deps.append("mobileclip")
        
        if missing_deps:
            error_msg = f"""
❌ Missing required dependencies: {', '.join(missing_deps)}

To install MobileCLIP, run:
    pip install git+https://github.com/apple/ml-mobileclip.git

For complete setup, run:
    bash setup_mobileclip.sh

Or install manually:
    conda create -n mobileclip_mobile python=3.10
    conda activate mobileclip_mobile
    pip install torch torchvision
    pip install git+https://github.com/apple/ml-mobileclip.git
    pip install coremltools tensorflow  # for mobile conversion
"""
            raise ImportError(error_msg)
    
    def load_pytorch_model(self, model_name: str, model_path: Optional[str] = None):
        """Load PyTorch MobileCLIP model."""
        if not MOBILECLIP_AVAILABLE:
            raise ImportError(f"MobileCLIP not available: {MOBILECLIP_IMPORT_ERROR}")
        
        try:
            logging.info(f"Loading MobileCLIP model: {model_name}")
            
            # Load model - handle both local files and model names
            if model_path and Path(model_path).exists():
                logging.info(f"Loading from local file: {model_path}")
                # Load from local file
                self.model, _, _ = mobileclip.create_model_and_transforms(
                    model_name, pretrained=model_path
                )
            else:
                logging.info(f"Loading pretrained model: {model_name}")
                # Try to load pretrained (this will download if needed)
                try:
                    self.model, _, _ = mobileclip.create_model_and_transforms(
                        model_name, pretrained=True
                    )
                except Exception as e:
                    logging.warning(f"Failed to load pretrained model: {e}")
                    logging.info("Loading model architecture without weights...")
                    self.model, _, _ = mobileclip.create_model_and_transforms(
                        model_name, pretrained=False
                    )
            
            # Reparameterize for inference
            self.model.eval()
            
            try:
                self.mobile_model = reparameterize_model(self.model)
                self.mobile_model.eval()
                logging.info("Model reparameterized for mobile inference")
            except Exception as e:
                logging.warning(f"Reparameterization failed: {e}")
                logging.info("Using original model without reparameterization")
                self.mobile_model = self.model
            
            logging.info(f"Successfully loaded {model_name}")
            
        except Exception as e:
            error_msg = f"""
❌ Failed to load MobileCLIP model '{model_name}': {e}

Troubleshooting steps:
1. Ensure MobileCLIP is installed:
   pip install git+https://github.com/apple/ml-mobileclip.git

2. Download the model first:
   python scripts/download/download_models.py --models {model_name}

3. Check available models:
   python -c "import mobileclip; print(mobileclip.list_models())"

4. Verify model file exists (if using local path):
   ls -la {model_path if model_path else 'models/pretrained/'}
"""
            logging.error(error_msg)
            raise RuntimeError(error_msg)
    
    def convert_to_coreml(self, output_dir: str, model_name: str = "mobileclip") -> str:
        """Convert model to CoreML format for iOS."""
        try:
            import coremltools as ct
        except ImportError:
            raise ImportError("""
❌ CoreML Tools not available. 

To install CoreML Tools (macOS only):
    pip install coremltools

Note: CoreML conversion only works on macOS.
For other platforms, use Android conversion instead.
""")
        
        if self.mobile_model is None:
            raise RuntimeError("No model loaded. Call load_pytorch_model() first.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Create example inputs
            example_image = torch.randn(1, 3, 224, 224)
            example_text = torch.randint(0, 1000, (1, 77))  # Text tokens
            
            logging.info("Tracing models for CoreML conversion...")
            
            # Trace the model components
            with torch.no_grad():
                # Check if model has separate encoders
                if hasattr(self.mobile_model, 'visual') and hasattr(self.mobile_model, 'text'):
                    # Split encoders
                    traced_image_model = torch.jit.trace(
                        self.mobile_model.visual, 
                        example_image
                    )
                    traced_text_model = torch.jit.trace(
                        self.mobile_model.text, 
                        example_text
                    )
                    
                    # Convert image encoder
                    logging.info("Converting image encoder to CoreML...")
                    image_model_path = output_path / f"{model_name}_image.mlpackage"
                    image_coreml_model = ct.convert(
                        traced_image_model,
                        inputs=[ct.ImageType(name="image", shape=example_image.shape)],
                        compute_units=ct.ComputeUnit.ALL,
                        minimum_deployment_target=ct.target.iOS15
                    )
                    image_coreml_model.save(str(image_model_path))
                    logging.info(f"Image encoder saved: {image_model_path}")
                    
                    # Convert text encoder  
                    logging.info("Converting text encoder to CoreML...")
                    text_model_path = output_path / f"{model_name}_text.mlpackage"
                    text_coreml_model = ct.convert(
                        traced_text_model,
                        inputs=[ct.TensorType(name="text_tokens", shape=example_text.shape)],
                        compute_units=ct.ComputeUnit.ALL,
                        minimum_deployment_target=ct.target.iOS15
                    )
                    text_coreml_model.save(str(text_model_path))
                    logging.info(f"Text encoder saved: {text_model_path}")
                    
                else:
                    # Convert entire model
                    logging.info("Converting complete model to CoreML...")
                    complete_model_path = output_path / f"{model_name}.mlpackage"
                    
                    # Create a wrapper for the complete model
                    class MobileCLIPWrapper(torch.nn.Module):
                        def __init__(self, model):
                            super().__init__()
                            self.model = model
                        
                        def forward(self, image, text):
                            image_features = self.model.encode_image(image)
                            text_features = self.model.encode_text(text)
                            return image_features, text_features
                    
                    wrapper = MobileCLIPWrapper(self.mobile_model)
                    traced_model = torch.jit.trace(wrapper, (example_image, example_text))
                    
                    coreml_model = ct.convert(
                        traced_model,
                        inputs=[
                            ct.ImageType(name="image", shape=example_image.shape),
                            ct.TensorType(name="text_tokens", shape=example_text.shape)
                        ],
                        compute_units=ct.ComputeUnit.ALL,
                        minimum_deployment_target=ct.target.iOS15
                    )
                    coreml_model.save(str(complete_model_path))
                    logging.info(f"Complete model saved: {complete_model_path}")
            
            logging.info(f"CoreML conversion completed: {output_path}")
            return str(output_path)
            
        except Exception as e:
            error_msg = f"""
❌ CoreML conversion failed: {e}

Troubleshooting:
1. Ensure you're on macOS with latest Xcode
2. Update CoreML Tools: pip install --upgrade coremltools
3. Check model compatibility
4. Try with a different model variant

Error details: {str(e)}
"""
            logging.error(error_msg)
            raise RuntimeError(error_msg)
    
    def convert_to_tflite(self, output_dir: str, model_name: str = "mobileclip") -> str:
        """Convert model to TensorFlow Lite format for Android."""
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("""
❌ TensorFlow not available.

To install TensorFlow:
    pip install tensorflow>=2.13.0

For GPU support:
    pip install tensorflow[and-cuda]
""")
        
        if self.mobile_model is None:
            raise RuntimeError("No model loaded. Call load_pytorch_model() first.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            logging.info("Converting to TensorFlow Lite via ONNX...")
            
            # Convert through ONNX first (more reliable)
            onnx_path = self._convert_to_onnx(output_path, model_name)
            
            # Convert ONNX to TensorFlow Lite
            tflite_path = self._onnx_to_tflite(onnx_path, output_path, model_name)
            
            logging.info(f"TensorFlow Lite model saved to {tflite_path}")
            return str(tflite_path)
            
        except Exception as e:
            error_msg = f"""
❌ TensorFlow Lite conversion failed: {e}

Troubleshooting:
1. Install required packages:
   pip install tensorflow onnx onnxruntime
   
2. For ONNX-TF conversion:
   pip install onnx-tf
   
3. Try converting without ONNX (direct PyTorch to TF):
   Set DIRECT_CONVERSION=True in config

Error details: {str(e)}
"""
            logging.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _convert_to_onnx(self, output_dir: Path, model_name: str) -> str:
        """Convert PyTorch model to ONNX."""
        try:
            import torch.onnx
        except ImportError:
            raise ImportError("ONNX not available. Install with: pip install onnx")
        
        # Example inputs
        example_image = torch.randn(1, 3, 224, 224)
        example_text = torch.randint(0, 1000, (1, 77))
        
        try:
            if hasattr(self.mobile_model, 'visual'):
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
                    output_names=['image_features'],
                    dynamic_axes={
                        'image': {0: 'batch_size'},
                        'image_features': {0: 'batch_size'}
                    }
                )
                logging.info(f"Image encoder ONNX saved: {image_onnx_path}")
                return str(image_onnx_path)
            else:
                # Export complete model
                complete_onnx_path = output_dir / f"{model_name}.onnx"
                
                # Create a simple wrapper
                class SimpleWrapper(torch.nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model
                    
                    def forward(self, image):
                        return self.model.encode_image(image)
                
                wrapper = SimpleWrapper(self.mobile_model)
                
                torch.onnx.export(
                    wrapper,
                    example_image,
                    str(complete_onnx_path),
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['image'],
                    output_names=['features']
                )
                logging.info(f"Complete model ONNX saved: {complete_onnx_path}")
                return str(complete_onnx_path)
                
        except Exception as e:
            logging.error(f"ONNX conversion failed: {e}")
            raise
    
    def _onnx_to_tflite(self, onnx_path: str, output_dir: Path, model_name: str) -> str:
        """Convert ONNX model to TensorFlow Lite."""
        try:
            import onnx
            import tensorflow as tf
        except ImportError:
            raise ImportError("Required packages missing: pip install onnx tensorflow")
        
        try:
            # Try onnx-tf if available
            try:
                from onnx_tf.backend import prepare
                
                # Load ONNX model
                onnx_model = onnx.load(onnx_path)
                
                # Convert to TensorFlow
                tf_rep = prepare(onnx_model)
                
                # Convert to TensorFlow Lite
                converter = tf.lite.TFLiteConverter.from_concrete_functions(tf_rep.signatures)
                
            except ImportError:
                logging.warning("onnx-tf not available, trying alternative conversion...")
                # Alternative: direct TensorFlow conversion
                # This is a simplified approach - may need adjustment
                raise ImportError("onnx-tf required for ONNX to TFLite conversion")
            
            # Optimization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
            # Convert
            tflite_model = converter.convert()
            
            # Save model
            tflite_path = output_dir / f"{model_name}.tflite"
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            return str(tflite_path)
            
        except Exception as e:
            logging.error(f"ONNX to TFLite conversion failed: {e}")
            
            # Fallback: create a placeholder file with instructions
            placeholder_path = output_dir / f"{model_name}_conversion_failed.txt"
            with open(placeholder_path, 'w') as f:
                f.write(f"""
TensorFlow Lite conversion failed: {e}

To complete Android conversion manually:

1. Install onnx-tf:
   pip install onnx-tf

2. Convert ONNX to TensorFlow:
   python -c "
   import onnx
   from onnx_tf.backend import prepare
   import tensorflow as tf
   
   onnx_model = onnx.load('{onnx_path}')
   tf_rep = prepare(onnx_model)
   
   converter = tf.lite.TFLiteConverter.from_concrete_functions(tf_rep.signatures)
   converter.optimizations = [tf.lite.Optimize.DEFAULT]
   tflite_model = converter.convert()
   
   with open('{model_name}.tflite', 'wb') as f:
       f.write(tflite_model)
   "

3. Alternative: Use Google Colab or online conversion tools

4. Check model compatibility with TensorFlow Lite
""")
            
            raise RuntimeError(f"TFLite conversion failed. See {placeholder_path} for manual steps.")
    
    def validate_conversion(self, original_output: torch.Tensor, converted_output: np.ndarray, tolerance: float = 1e-3) -> bool:
        """Validate model conversion accuracy."""
        try:
            original_np = original_output.detach().cpu().numpy()
            
            # Compare outputs
            diff = np.abs(original_np - converted_output)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            logging.info(f"Conversion validation - Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
            
            return max_diff < tolerance
        except Exception as e:
            logging.warning(f"Validation failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if self.mobile_model is None:
            return {"error": "No model loaded"}
        
        try:
            total_params = sum(p.numel() for p in self.mobile_model.parameters())
            trainable_params = sum(p.numel() for p in self.mobile_model.parameters() if p.requires_grad)
            
            return {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "model_type": type(self.mobile_model).__name__,
                "has_visual_encoder": hasattr(self.mobile_model, 'visual'),
                "has_text_encoder": hasattr(self.mobile_model, 'text'),
                "input_resolution": [224, 224],
                "text_context_length": 77
            }
        except Exception as e:
            return {"error": f"Failed to get model info: {e}"}


# Test function to verify installation
def test_mobileclip_installation():
    """Test MobileCLIP installation and basic functionality."""
    print("🧪 Testing MobileCLIP installation...")
    
    try:
        import mobileclip
        print("✅ MobileCLIP imported successfully")
        
        # Test model creation
        model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained=False)
        print("✅ Model creation successful")
        
        tokenizer = mobileclip.get_tokenizer('mobileclip_s0')
        print("✅ Tokenizer creation successful")
        
        # Test basic functionality
        test_image = torch.randn(1, 3, 224, 224)
        test_text = tokenizer(["test"])
        
        with torch.no_grad():
            image_features = model.encode_image(test_image)
            text_features = model.encode_text(test_text)
        
        print(f"✅ Model inference successful")
        print(f"   Image features shape: {image_features.shape}")
        print(f"   Text features shape: {text_features.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ MobileCLIP test failed: {e}")
        return False


if __name__ == "__main__":
    # Run test if script is executed directly
    success = test_mobileclip_installation()
    sys.exit(0 if success else 1)
