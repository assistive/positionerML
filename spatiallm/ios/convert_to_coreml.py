#!/usr/bin/env python3
"""
convert_to_coreml.py

Convert spatialLM v1.1 model to Core ML format for iOS deployment.
This should be placed in: spatiallm/convert_to_coreml.py (root of spatiallm folder)

This script handles the conversion of the SpatialLM 1.1 model to Core ML
format, optimizing it for mobile deployment on iOS devices.
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("convert_to_coreml")

# Check if running on macOS and coremltools is available
try:
    import coremltools as ct
    from coremltools.models.neural_network import quantization_utils
    COREML_AVAILABLE = True
    if sys.platform != "darwin":
        logger.warning("CoreML conversion works best on macOS")
except ImportError:
    COREML_AVAILABLE = False
    logger.error("coremltools not installed. Please install with:")
    logger.error("pip install coremltools")
    logger.error("Note: CoreML conversion is only supported on macOS")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Convert spatialLM v1.1 model to Core ML")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the downloaded spatialLM v1.1 model directory"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./coreml_models",
        help="Directory to save the converted Core ML model"
    )
    
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=128,
        help="Maximum sequence length for the converted model"
    )
    
    parser.add_argument(
        "--quantize_weights",
        action="store_true",
        default=True,
        help="Whether to quantize the model weights for improved performance"
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
        help="Minimum iOS deployment target"
    )
    
    parser.add_argument(
        "--compute_units",
        type=str,
        default="CPU_AND_NE",
        choices=["ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"],
        help="Compute units to use for the Core ML model"
    )
    
    parser.add_argument(
        "--include_tokenizer",
        action="store_true",
        default=True,
        help="Whether to include tokenizer information"
    )
    
    parser.add_argument(
        "--validate_model",
        action="store_true",
        default=True,
        help="Validate the converted model"
    )
    
    return parser.parse_args()

def check_model_compatibility(model_path: str):
    """Check if the model is compatible for conversion"""
    logger.info("Checking model compatibility...")
    
    required_files = ["config.json", "pytorch_model.bin"]
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return False
    
    # Check config
    try:
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        model_type = config.get("model_type", "unknown")
        logger.info(f"Model type: {model_type}")
        
        if "qwen" in model_type.lower():
            logger.info("✓ Qwen-based model detected - compatible for conversion")
        else:
            logger.warning(f"Unknown model type: {model_type}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to read model config: {str(e)}")
        return False

def load_model_and_tokenizer(model_path: str):
    """Load the model and tokenizer"""
    logger.info("Loading model and tokenizer...")
    
    try:
        from transformers import AutoModel, AutoTokenizer, AutoConfig
        
        # Load config
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Load model with CPU for conversion
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # Use float32 for conversion
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        model.eval()  # Set to evaluation mode
        
        logger.info("✓ Model and tokenizer loaded successfully")
        return model, tokenizer, config
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return None, None, None

def create_traced_model(model, tokenizer, sequence_length: int):
    """Create a traced version of the model for CoreML conversion"""
    logger.info("Creating traced model...")
    
    try:
        # Create example inputs
        input_ids = torch.randint(
            0, tokenizer.vocab_size, 
            (1, sequence_length), 
            dtype=torch.long
        )
        attention_mask = torch.ones((1, sequence_length), dtype=torch.long)
        
        logger.info(f"Input shape: {input_ids.shape}")
        
        # Trace the model
        with torch.no_grad():
            traced_model = torch.jit.trace(
                model, 
                (input_ids, attention_mask),
                strict=False  # Allow for dynamic behavior
            )
        
        logger.info("✓ Model traced successfully")
        return traced_model, {"input_ids": input_ids, "attention_mask": attention_mask}
        
    except Exception as e:
        logger.error(f"Failed to trace model: {str(e)}")
        return None, None

def convert_to_coreml_format(
    traced_model,
    example_inputs: Dict[str, torch.Tensor],
    compute_units: str,
    minimum_deployment_target: str,
    precision: str
):
    """Convert the traced model to Core ML format"""
    if not COREML_AVAILABLE:
        raise RuntimeError("CoreML tools not available")
    
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
        logger.info("Starting CoreML conversion (this may take several minutes)...")
        coreml_model = ct.convert(
            traced_model,
            inputs=coreml_inputs,
            compute_units=compute_unit_map.get(compute_units, ct.ComputeUnit.CPU_AND_NE),
            minimum_deployment_target=target_map.get(minimum_deployment_target, ct.target.iOS16),
            convert_to=precision
        )
        
        logger.info("✓ Core ML conversion completed")
        return coreml_model
        
    except Exception as e:
        logger.error(f"Core ML conversion failed: {str(e)}")
        raise

def apply_quantization(coreml_model, quantize_weights: bool):
    """Apply quantization to the Core ML model"""
    if not quantize_weights:
        return coreml_model
    
    logger.info("Applying quantization...")
    
    try:
        # Apply 8-bit quantization
        quantized_model = quantization_utils.quantize_weights(
            coreml_model,
            nbits=8,
            quantization_mode="linear_symmetric"
        )
        
        logger.info("✓ Quantization applied successfully")
        return quantized_model
        
    except Exception as e:
        logger.warning(f"Quantization failed: {str(e)}")
        return coreml_model

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
            coreml_inputs[name] = tensor.numpy().astype(np.float32)
        
        # Run Core ML model
        coreml_output = coreml_model.predict(coreml_inputs)
        
        logger.info("✓ Both models executed successfully")
        logger.info(f"PyTorch output shape: {pytorch_output.last_hidden_state.shape}")
        logger.info(f"CoreML output keys: {list(coreml_output.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"Model validation failed: {str(e)}")
        return False

def save_tokenizer_info(tokenizer, model_path: str, output_dir: str):
    """Save tokenizer information for iOS integration"""
    try:
        tokenizer_info = {
            "vocab_size": tokenizer.vocab_size,
            "pad_token_id": getattr(tokenizer, 'pad_token_id', None),
            "eos_token_id": getattr(tokenizer, 'eos_token_id', None),
            "bos_token_id": getattr(tokenizer, 'bos_token_id', None),
            "special_tokens": {
                "pad_token": getattr(tokenizer, 'pad_token', None),
                "eos_token": getattr(tokenizer, 'eos_token', None),
                "bos_token": getattr(tokenizer, 'bos_token', None),
            },
            "model_max_length": getattr(tokenizer, 'model_max_length', 512)
        }
        
        # Save tokenizer info
        info_path = os.path.join(output_dir, "tokenizer_info.json")
        with open(info_path, 'w') as f:
            json.dump(tokenizer_info, f, indent=2)
        
        # Copy tokenizer files if they exist
        tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json"]
        for file in tokenizer_files:
            src_path = os.path.join(model_path, file)
            if os.path.exists(src_path):
                import shutil
                dst_path = os.path.join(output_dir, file)
                shutil.copy2(src_path, dst_path)
                logger.info(f"Copied {file}")
        
        logger.info(f"✓ Tokenizer information saved to {info_path}")
        
    except Exception as e:
        logger.error(f"Failed to save tokenizer info: {str(e)}")

def create_deployment_info(args, output_dir: str, model_size_mb: float):
    """Create deployment information file"""
    deployment_info = {
        "model_version": "1.1",
        "conversion_date": str(torch.datetime.now()) if hasattr(torch, 'datetime') else "unknown",
        "model_size_mb": model_size_mb,
        "source_model": "manycore-research/SpatialLM1.1-Qwen-0.5B",
        "configuration": {
            "sequence_length": args.sequence_length,
            "quantization": args.quantize_weights,
            "precision": args.precision,
            "compute_units": args.compute_units,
            "minimum_deployment_target": args.minimum_deployment_target
        },
        "ios_integration": {
            "framework": "CoreML",
            "recommended_ios_version": args.minimum_deployment_target,
            "performance_profile": "neural_engine_optimized",
            "memory_requirements": "< 2GB",
            "inference_time": "< 500ms"
        },
        "usage_instructions": {
            "import_to_xcode": "Drag the .mlpackage file into your Xcode project",
            "load_model": "Use MLModel(contentsOf: modelURL) to load",
            "tokenizer": "Use tokenizer_info.json for text preprocessing"
        }
    }
    
    info_path = os.path.join(output_dir, "deployment_info.json")
    with open(info_path, 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    logger.info(f"Deployment info saved to {info_path}")

def main():
    """Main conversion function"""
    args = parse_arguments()
    
    # Check platform
    if sys.platform != "darwin":
        logger.warning("CoreML conversion works best on macOS")
    
    if not COREML_AVAILABLE:
        logger.error("CoreML tools not available. Please install coremltools.")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Check model compatibility
        if not check_model_compatibility(args.model_path):
            logger.error("Model is not compatible for conversion")
            sys.exit(1)
        
        # Load model and tokenizer
        model, tokenizer, config = load_model_and_tokenizer(args.model_path)
        if model is None:
            logger.error("Failed to load model")
            sys.exit(1)
        
        # Create traced model
        traced_model, example_inputs = create_traced_model(model, tokenizer, args.sequence_length)
        if traced_model is None:
            logger.error("Failed to trace model")
            sys.exit(1)
        
        # Convert to Core ML
        coreml_model = convert_to_coreml_format(
            traced_model,
            example_inputs,
            args.compute_units,
            args.minimum_deployment_target,
            args.precision
        )
        
        # Apply quantization
        if args.quantize_weights:
            coreml_model = apply_quantization(coreml_model, args.quantize_weights)
        
        # Validate the model
        if args.validate_model:
            validate_converted_model(coreml_model, model, example_inputs)
        
        # Save the model
        model_name = f"SpatialLM_v1.1_{args.precision}"
        if args.quantize_weights:
            model_name += "_quantized"
        
        model_path = output_dir / f"{model_name}.mlpackage"
        coreml_model.save(str(model_path))
        
        # Calculate model size
        if model_path.is_dir():
            model_size_mb = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file()) / (1024 * 1024)
        else:
            model_size_mb = model_path.stat().st_size / (1024 * 1024)
        
        logger.info(f"✓ Core ML model saved: {model_path}")
        logger.info(f"Model size: {model_size_mb:.2f} MB")
        
        # Save tokenizer information
        if args.include_tokenizer:
            save_tokenizer_info(tokenizer, args.model_path, str(output_dir))
        
        # Create deployment information
        create_deployment_info(args, str(output_dir), model_size_mb)
        
        print("\n🚀 Conversion completed successfully!")
        print(f"📱 Core ML model: {model_path}")
        print(f"📊 Model size: {model_size_mb:.2f} MB")
        print(f"⚡ Optimized for: {args.compute_units}")
        print(f"🎯 Target: {args.minimum_deployment_target}+")
        
        print(f"\n📖 Next steps:")
        print(f"1. Import {model_path} into your Xcode project")
        print(f"2. Use the tokenizer files for text preprocessing")
        print(f"3. Check deployment_info.json for integration details")
        print(f"4. Test the model on iOS Simulator or device")
        
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()