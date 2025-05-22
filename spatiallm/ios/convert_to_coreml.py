#!/usr/bin/env python3
"""
Convert spatialLM model to Core ML format for iOS deployment.

This script handles the conversion of a trained spatialLM model to Core ML
format, optimizing it for mobile deployment on iOS devices.
"""

import os
import sys
import logging
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformers import AutoTokenizer
from models.spatialLM import SpatialLM

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("convert_to_coreml")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Convert spatialLM model to Core ML")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained spatialLM model"
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
        help="Whether to quantize the model weights for improved performance and smaller size"
    )
    
    parser.add_argument(
        "--precision",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Precision for the Core ML model"
    )
    
    parser.add_argument(
        "--minimum_deployment_target",
        type=str,
        default="iOS15",
        choices=["iOS13", "iOS14", "iOS15", "iOS16", "iOS17"],
        help="Minimum iOS deployment target"
    )
    
    parser.add_argument(
        "--include_tokenizer",
        action="store_true",
        help="Whether to include the tokenizer in the output directory"
    )
    
    parser.add_argument(
        "--optimize_for",
        type=str,
        default="neuralengine",
        choices=["neuralengine", "cpuandgpu", "cpuonly"],
        help="Optimization target for the Core ML model"
    )
    
    parser.add_argument(
        "--compute_units",
        type=str,
        default="ALL",
        choices=["ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"],
        help="Compute units to use for the Core ML model"
    )
    
    return parser.parse_args()

def load_pytorch_model(model_path):
    """
    Load the PyTorch spatialLM model.
    
    Args:
        model_path: Path to the trained model
        
    Returns:
        model: The loaded model
        tokenizer: The tokenizer
    """
    logger.info(f"Loading model from {model_path}")
    
    try:
        # Load model
        model = SpatialLM.from_pretrained(model_path)
        model.eval()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def create_trace_model(model, batch_size=1, sequence_length=64, spatial_dim=3):
    """
    Create a traced version of the model for export.
    
    Args:
        model: PyTorch model
        batch_size: Batch size for the converted model
        sequence_length: Sequence length for the converted model
        spatial_dim: Number of spatial dimensions
        
    Returns:
        traced_model: Traced PyTorch model
    """
    logger.info("Creating traced model for export")
    
    # Create dummy inputs
    dummy_input_ids = torch.ones((batch_size, sequence_length), dtype=torch.long)
    dummy_attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.long)
    dummy_spatial_coords = torch.zeros((batch_size, spatial_dim), dtype=torch.float)
    
    # Define a wrapper class for tracing
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, input_ids, attention_mask, spatial_coordinates):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                spatial_coordinates=spatial_coordinates,
                return_dict=True
            )
            return outputs.logits, outputs.spatial_predictions
    
    # Create wrapper
    wrapper = ModelWrapper(model)
    
    # Trace model
    with torch.no_grad():
        traced_model = torch.jit.trace(
            wrapper,
            (dummy_input_ids, dummy_attention_mask, dummy_spatial_coords)
        )
    
    return traced_model

def convert_to_coreml(model, output_path, args):
    """
    Convert PyTorch model to Core ML format.
    
    Args:
        model: PyTorch model
        output_path: Path to save the Core ML model
        args: Command line arguments
        
    Returns:
        mlmodel_path: Path to the saved Core ML model
    """
    logger.info("Converting PyTorch model to Core ML format")
    
    # Import coremltools
    try:
        import coremltools as ct
    except ImportError:
        logger.error("coremltools not installed. Please install it with:")
        logger.error("pip install -U coremltools")
        sys.exit(1)
    
    # Create traced model
    traced_model = create_trace_model(
        model,
        args.batch_size,
        args.sequence_length,
        args.spatial_dim
    )
    
    # Define input and output shapes
    input_shapes = {
        "input_ids": (args.batch_size, args.sequence_length),
        "attention_mask": (args.batch_size, args.sequence_length),
        "spatial_coordinates": (args.batch_size, args.spatial_dim)
    }
    
    # Convert to Core ML
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="input_ids", shape=input_shapes["input_ids"], dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=input_shapes["attention_mask"], dtype=np.int32),
            ct.TensorType(name="spatial_coordinates", shape=input_shapes["spatial_coordinates"], dtype=np.float32)
        ],
        outputs=[
            ct.TensorType(name="logits"),
            ct.TensorType(name="spatial_predictions")
        ],
        minimum_deployment_target=getattr(ct.target, args.minimum_deployment_target),
        convert_to="mlprogram"  # Use the newer ML Program format
    )
    
    # Set compute units
    compute_units_map = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE
    }
    
    mlmodel.compute_unit = compute_units_map[args.compute_units]
    
    # Set model metadata
    mlmodel.author = "spatialLM"
    mlmodel.license = "MIT"
    mlmodel.version = "1.0"
    mlmodel.short_description = "spatialLM model for spatial language understanding"
    
    # Quantize weights if requested
    if args.quantize_weights:
        logger.info("Quantizing model weights")
        mlmodel = ct.models.neural_network.quantization_utils.quantize_weights(mlmodel, nbits=8)
    
    # Convert to fp16 if requested
    if args.precision == "float16":
        logger.info("Converting to float16 precision")
        mlmodel = ct.models.neural_network.quantization_utils.quantize_weights(mlmodel, dtype=np.float16)
    
    # Save model
    mlmodel_path = os.path.join(output_path, "spatialLM_model.mlmodel")
    mlmodel.save(mlmodel_path)
    
    logger.info(f"Core ML model saved to {mlmodel_path}")
    return mlmodel_path

def save_model_metadata(model, tokenizer, output_path, args):
    """
    Save model metadata for use in iOS app.
    
    Args:
        model: The model
        tokenizer: The tokenizer
        output_path: Path to save metadata
        args: Command line arguments
    """
    logger.info("Saving model metadata")
    
    # Create metadata
    metadata = {
        "model_type": "spatialLM",
        "base_model_name": model.config.base_model_name,
        "spatial_dim": model.config.spatial_dim,
        "sequence_length": args.sequence_length,
        "vocab_size": len(tokenizer),
        "tokenizer_type": tokenizer.__class__.__name__,
        "quantized_weights": args.quantize_weights,
        "precision": args.precision,
        "special_tokens": {
            "pad_token": tokenizer.pad_token,
            "eos_token": tokenizer.eos_token,
            "bos_token": tokenizer.bos_token if hasattr(tokenizer, "bos_token") else None,
            "unk_token": tokenizer.unk_token,
        }
    }
    
    # Save metadata
    metadata_path = os.path.join(output_path, "model_metadata.json")
    with open(metadata_path, "w") as f:
        import json
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Model metadata saved to {metadata_path}")

def prepare_tokenizer_for_ios(tokenizer, output_path):
    """
    Prepare tokenizer files for iOS integration.
    
    Args:
        tokenizer: The tokenizer
        output_path: Path to save tokenizer files
    """
    tokenizer_dir = os.path.join(output_path, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)
    
    # Save tokenizer
    tokenizer.save_pretrained(tokenizer_dir)
    
    # Create a simplified vocabulary file for iOS
    vocab = tokenizer.get_vocab()
    
    # Sort vocabulary by token id
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    
    # Save as a plain text file, one token per line
    vocab_path = os.path.join(tokenizer_dir, "vocab.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        for token, _ in sorted_vocab:
            f.write(f"{token}\n")
    
    logger.info(f"Tokenizer prepared for iOS and saved to {tokenizer_dir}")

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load PyTorch model
    model, tokenizer = load_pytorch_model(args.model_path)
    
    # Convert to Core ML
    mlmodel_path = convert_to_coreml(model, args.output_dir, args)
    
    # Save model metadata
    save_model_metadata(model, tokenizer, args.output_dir, args)
    
    # Save tokenizer if requested
    if args.include_tokenizer:
        prepare_tokenizer_for_ios(tokenizer, args.output_dir)
    
    logger.info("Conversion completed successfully!")
    logger.info(f"Core ML model saved to {mlmodel_path}")

if __name__ == "__main__":
    main()
