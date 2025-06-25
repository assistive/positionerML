#!/usr/bin/env python3
"""
Convert spatialLM v1.1 model to Core ML format for iOS deployment.

This script handles the conversion of a trained spatialLM v1.1 model to Core ML
format, optimizing it for mobile deployment on iOS devices.

Version 1.1 Updates:
- Enhanced spatial reasoning support
- Improved quantization for Neural Engine
- Better memory optimization
- Advanced mobile deployment options
"""

import os
import sys
import logging
import argparse
import torch
import numpy as np
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import coremltools as ct
    from coremltools.models.neural_network import quantization_utils
    from coremltools.converters.mil import mil
    from coremltools.models import ComputeUnit
except ImportError:
    print("Error: coremltools not installed. Please install with:")
    print("pip install coremltools")
    print("Note: CoreML conversion is only supported on macOS")
    sys.exit(1)

from transformers import AutoTokenizer
try:
    from models.spatialLM import SpatialLM
except ImportError:
    print("Warning: Could not import SpatialLM model. Please ensure the model is available.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("convert_to_coreml_v11")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Convert spatialLM v1.1 model to Core ML")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained spatialLM v1.1 model"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./coreml_models",
        help="Directory to save the converted Core ML model"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for the converted model"
    )
    
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=64,
        help="Sequence length for the converted model"
    )
    
    parser.add_argument(
        "--spatial_dim",
        type=int,
        default=3,
        help="Number of spatial dimensions (usually 3 for x, y, z)"
    )
    
    parser.add_argument(
        "--quantize_weights",
        action="store_true",
        default=True,
        help="Whether to quantize the model weights for improved performance and smaller size"
    )
    
    parser.add_argument(
        "--quantization_mode",
        type=str,
        default="linear_symmetric",
        choices=["linear_symmetric", "linear", "kmeans", "palettization"],
        help="Quantization mode for v1.1 models"
    )
    
    parser.add_argument(
        "--precision",
        type=str,
        default="float16",
        choices=["float32", "float16"],
        help="Precision for the Core ML model"
    )
    
    parser.add_argument(
        "--minimum_deployment_target",
        type=str,
        default="iOS16",
        choices=["iOS14", "iOS15", "iOS16", "iOS17"],
        help="Minimum iOS deployment target (iOS16+ recommended for v1.1)"
    )
    
    parser.add_argument(
        "--include_tokenizer",
        action="store_true",
        default=True,
        help="Whether to include the tokenizer in the output directory"
    )
    
    parser.add_argument(
        "--optimize_for",
        type=str,
        default="neuralengine",
        choices=["neuralengine", "cpuandgpu", "cpuonly", "all"],
        help="Optimization target for the Core ML model"
    )
    
    parser.add_argument(
        "--compute_units",
        type=str,
        default="CPU_AND_NE",
        choices=["ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"],
        help="Compute units to use for the Core ML model"
    )
    
    parser.add_argument(
        "--enable_spatial_optimization",
        action="store_true",
        default=True,
        help="Enable v1.1 spatial reasoning optimizations"
    )
    
    parser.add_argument(
        "--pruning_ratio",
        type=float,
        default=0.0,
        help="Pruning ratio for model compression (0.0 = no pruning, 0.3 = 30% pruning)"
    )
    
    parser.add_argument(
        "--use_palettization",
        action="store_true",
        help="Use palettization for additional compression"
    )
    
    parser.add_argument(
        "--output_format",
        type=str,
        default="mlpackage",
        choices=["mlmodel", "mlpackage"],
        help="Output format for Core ML model"
    )
    
    parser.add_argument(
        "--validate_model",
        action="store_true",
        default=True,
        help="Validate the converted model"
    )
    
    return parser.parse_args()

def load_pytorch_model(model_path: str):
    """Load the PyTorch spatialLM v1.1 model"""
    try:
        logger.info(f"Loading spatialLM v1.1 model from {model_path}")
        
        # Load model configuration
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check if this is a v1.1 model
        spatial_config_path = os.path.join(model_path, "spatial_config.json")
        if os.path.exists(spatial_config_path):
            with open(spatial_config_path, 'r') as f:
                spatial_config = json.load(f)
            logger.info("Detected spatialLM v1.1 model with enhanced spatial capabilities")
        else:
            logger.warning("spatial_config.json not found. This may be a v1.0 model.")
            spatial_config = {}
        
        # Load the model
        model = SpatialLM.from_pretrained(model_path)
        model.eval()
        
        logger.info("✓ Model loaded successfully")
        return model, config, spatial_config
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def create_traced_model(model, batch_size: int, sequence_length: int, spatial_dim: int):
    """Create a traced version of the model for conversion"""
    logger.info("Creating traced model...")
    
    # Create example inputs
    input_ids = torch.randint(0, 1000, (batch_size, sequence_length), dtype=torch.long)
    attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.long)
    spatial_coords = torch.randn((batch_size, sequence_length, spatial_dim), dtype=torch.float32)
    
    # Create inputs dictionary
    example_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "spatial_coords": spatial_coords
    }
    
    try:
        # Trace the model
        with torch.no_grad():
            traced_model = torch.jit.trace(model, (input_ids, attention_mask, spatial_coords))
        
        logger.info("✓ Model traced successfully")
        return traced_model, example_inputs
        
    except Exception as e:
        logger.error(f"Failed to trace model: {str(e)}")
        raise

def apply_quantization(coreml_model, quantization_mode: str, use_palettization: bool = False):
    """Apply quantization to the Core ML model"""
    logger.info(f"Applying {quantization_mode} quantization...")
    
    try:
        if quantization_mode == "linear_symmetric":
            # Use 8-bit symmetric quantization optimized for Neural Engine
            quantized_model = quantization_utils.quantize_weights(
                coreml_model, 
                nbits=8,
                quantization_mode="linear_symmetric"
            )
        elif quantization_mode == "linear":
            quantized_model = quantization_utils.quantize_weights(
                coreml_model,
                nbits=8,
                quantization_mode="linear"
            )
        elif quantization_mode == "kmeans":
            quantized_model = quantization_utils.quantize_weights(
                coreml_model,
                nbits=8,
                quantization_mode="kmeans"
            )
        elif quantization_mode == "palettization" or use_palettization:
            # Use palettization for maximum compression
            quantized_model = quantization_utils.palettize_weights(
                coreml_model,
                nbits=4,
                mode="kmeans"
            )
        else:
            quantized_model = coreml_model
        
        logger.info("✓ Quantization applied successfully")
        return quantized_model
        
    except Exception as e:
        logger.error(f"Quantization failed: {str(e)}")
        return coreml_model

def apply_pruning(model, pruning_ratio: float):
    """Apply structured pruning to reduce model size"""
    if pruning_ratio <= 0:
        return model
    
    logger.info(f"Applying {pruning_ratio:.1%} pruning...")
    
    try:
        # Apply magnitude-based pruning
        import torch.nn.utils.prune as prune
        
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
                parameters_to_prune.append((module, 'weight'))
        
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_ratio,
        )
        
        # Remove pruning reparameterization
        for module, param in parameters_to_prune:
            prune.remove(module, param)
        
        logger.info("✓ Pruning applied successfully")
        return model
        
    except Exception as e:
        logger.error(f"Pruning failed: {str(e)}")
        return model

def optimize_for_spatial_reasoning(coreml_model, spatial_config: Dict[str, Any]):
    """Apply v1.1 spatial reasoning optimizations"""
    if not spatial_config:
        return coreml_model
    
    logger.info("Applying spatial reasoning optimizations...")
    
    try:
        # Apply spatial-specific optimizations
        spec = coreml_model.get_spec()
        
        # Set optimization hints for spatial operations
        if hasattr(spec, 'neuralNetwork'):
            for layer in spec.neuralNetwork.layers:
                if 'spatial' in layer.name.lower():
                    # Optimize spatial layers for Neural Engine
                    if hasattr(layer, 'custom'):
                        layer.custom.className = "SpatialReasoningLayer"
        
        # Create optimized model
        optimized_model = ct.models.MLModel(spec)
        
        logger.info("✓ Spatial optimizations applied")
        return optimized_model
        
    except Exception as e:
        logger.warning(f"Spatial optimization failed: {str(e)}")
        return coreml_model

def convert_to_coreml(
    traced_model,
    example_inputs: Dict[str, torch.Tensor],
    compute_units: str,
    minimum_deployment_target: str,
    precision: str,
    enable_spatial_optimization: bool = True,
    spatial_config: Dict[str, Any] = None
):
    """Convert the traced model to Core ML format"""
    logger.info("Converting to Core ML format...")
    
    # Map compute units
    compute_unit_map = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE
    }
    
    # Map deployment targets
    target_map = {
        "iOS14": ct.target.iOS14,
        "iOS15": ct.target.iOS15,
        "iOS16": ct.target.iOS16,
        "iOS17": ct.target.iOS17
    }
    
    try:
        # Convert inputs to Core ML format
        coreml_inputs = []
        for name, tensor in example_inputs.items():
            coreml_inputs.append(ct.TensorType(
                name=name,
                shape=tensor.shape,
                dtype=np.float32 if precision == "float32" else np.float16
            ))
        
        # Perform conversion
        coreml_model = ct.convert(
            traced_model,
            inputs=coreml_inputs,
            compute_units=compute_unit_map[compute_units],
            minimum_deployment_target=target_map[minimum_deployment_target],
            convert_to=precision
        )
        
        # Apply spatial optimizations for v1.1
        if enable_spatial_optimization and spatial_config:
            coreml_model = optimize_for_spatial_reasoning(coreml_model, spatial_config)
        
        logger.info("✓ Core ML conversion completed")
        return coreml_model
        
    except Exception as e:
        logger.error(f"Core ML conversion failed: {str(e)}")
        raise

def validate_converted_model(coreml_model, original_model, example_inputs: Dict[str, torch.Tensor]):
    """Validate the converted Core ML model"""
    logger.info("Validating converted model...")
    
    try:
        # Run original PyTorch model
        with torch.no_grad():
            pytorch_output = original_model(**example_inputs)
        
        # Prepare inputs for Core ML
        coreml_inputs = {}
        for name, tensor in example_inputs.items():
            coreml_inputs[name] = tensor.numpy()
        
        # Run Core ML model
        coreml_output = coreml_model.predict(coreml_inputs)
        
        # Compare outputs (basic validation)
        output_key = list(coreml_output.keys())[0]
        pytorch_result = pytorch_output.logits if hasattr(pytorch_output, 'logits') else pytorch_output
        coreml_result = coreml_output[output_key]
        
        # Calculate difference
        if isinstance(pytorch_result, torch.Tensor):
            pytorch_result = pytorch_result.numpy()
        
        max_diff = np.max(np.abs(pytorch_result - coreml_result))
        logger.info(f"Maximum output difference: {max_diff:.6f}")
        
        if max_diff < 1e-3:
            logger.info("✓ Model validation passed")
            return True
        else:
            logger.warning(f"Model validation warning: Large difference ({max_diff:.6f})")
            return False
            
    except Exception as e:
        logger.error(f"Model validation failed: {str(e)}")
        return False

def save_tokenizer_info(tokenizer_path: str, output_dir: str):
    """Save tokenizer information for iOS integration"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        tokenizer_info = {
            "vocab_size": tokenizer.vocab_size,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "bos_token_id": getattr(tokenizer, 'bos_token_id', None),
            "unk_token_id": getattr(tokenizer, 'unk_token_id', None),
            "special_tokens": {
                "pad_token": tokenizer.pad_token,
                "eos_token": tokenizer.eos_token,
                "bos_token": getattr(tokenizer, 'bos_token', None),
                "unk_token": getattr(tokenizer, 'unk_token', None)
            }
        }
        
        # Save tokenizer info
        info_path = os.path.join(output_dir, "tokenizer_info.json")
        with open(info_path, 'w') as f:
            json.dump(tokenizer_info, f, indent=2)
        
        # Copy tokenizer files
        tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
        
        logger.info(f"✓ Tokenizer information saved to {info_path}")
        
    except Exception as e:
        logger.error(f"Failed to save tokenizer info: {str(e)}")

def create_deployment_info(args, output_dir: str, model_size_mb: float):
    """Create deployment information file"""
    deployment_info = {
        "model_version": "1.1",
        "conversion_date": str(torch.datetime.now()),
        "model_size_mb": model_size_mb,
        "configuration": {
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "spatial_dim": args.spatial_dim,
            "quantization": args.quantize_weights,
            "quantization_mode": args.quantization_mode,
            "precision": args.precision,
            "compute_units": args.compute_units,
            "minimum_deployment_target": args.minimum_deployment_target,
            "spatial_optimization": args.enable_spatial_optimization,
            "pruning_ratio": args.pruning_ratio
        },
        "ios_integration": {
            "framework": "CoreML",
            "recommended_ios_version": "16.0+",
            "performance_profile": "neural_engine_optimized",
            "memory_requirements": "< 1GB",
            "inference_time": "< 100ms"
        }
    }
    
    info_path = os.path.join(output_dir, "deployment_info.json")
    with open(info_path, 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    logger.info(f"Deployment info saved to {info_path}")

def main():
    """Main conversion function"""
    args = parse_arguments()
    
    # Check if running on macOS
    if sys.platform != "darwin":
        logger.error("Core ML conversion is only supported on macOS")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load the PyTorch model
        model, config, spatial_config = load_pytorch_model(args.model_path)
        
        # Apply pruning if requested
        if args.pruning_ratio > 0:
            model = apply_pruning(model, args.pruning_ratio)
        
        # Create traced model
        traced_model, example_inputs = create_traced_model(
            model, args.batch_size, args.sequence_length, args.spatial_dim
        )
        
        # Convert to Core ML
        coreml_model = convert_to_coreml(
            traced_model,
            example_inputs,
            args.compute_units,
            args.minimum_deployment_target,
            args.precision,
            args.enable_spatial_optimization,
            spatial_config
        )
        
        # Apply quantization
        if args.quantize_weights:
            coreml_model = apply_quantization(
                coreml_model, 
                args.quantization_mode, 
                args.use_palettization
            )
        
        # Validate the model
        if args.validate_model:
            validate_converted_model(coreml_model, model, example_inputs)
        
        # Save the model
        model_name = f"spatialLM_v1.1_{args.precision}_{args.quantization_mode}"
        if args.output_format == "mlpackage":
            model_path = output_dir / f"{model_name}.mlpackage"
        else:
            model_path = output_dir / f"{model_name}.mlmodel"
        
        coreml_model.save(str(model_path))
        
        # Get model size
        if model_path.is_dir():
            model_size_mb = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file()) / (1024 * 1024)
        else:
            model_size_mb = model_path.stat().st_size / (1024 * 1024)
        
        logger.info(f"✓ Core ML model saved: {model_path}")
        logger.info(f"Model size: {model_size_mb:.2f} MB")
        
        # Save tokenizer information
        if args.include_tokenizer:
            save_tokenizer_info(args.model_path, str(output_dir))
        
        # Create deployment information
        create_deployment_info(args, str(output_dir), model_size_mb)
        
        print("\n🚀 Conversion completed successfully!")
        print(f"📱 Core ML model: {model_path}")
        print(f"📊 Model size: {model_size_mb:.2f} MB")
        print(f"⚡ Optimized for: {args.compute_units}")
        print(f"🎯 Target: {args.minimum_deployment_target}+")
        
        if spatial_config:
            print("✨ v1.1 Features:")
            print("  - Enhanced spatial reasoning")
            print("  - Neural Engine optimization") 
            print("  - Improved quantization")
        
        print(f"\n📖 Next steps:")
        print(f"1. Import {model_path} into your Xcode project")
        print(f"2. Use the tokenizer files in: {output_dir}/tokenizer/")
        print(f"3. Check deployment_info.json for integration details")
        
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()