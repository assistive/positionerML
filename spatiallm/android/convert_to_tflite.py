#!/usr/bin/env python3
"""
Convert spatialLM model to TensorFlow Lite format for Android deployment.

This script handles the conversion of a trained spatialLM model to TensorFlow Lite
format, optimizing it for mobile deployment on Android.
"""

import os
import sys
import logging
import argparse
import torch
import tensorflow as tf
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
logger = logging.getLogger("convert_to_tflite")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Convert spatialLM model to TensorFlow Lite")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained spatialLM model"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tflite_models",
        help="Directory to save the converted TFLite model"
    )
    
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Whether to quantize the model for improved performance and smaller size"
    )
    
    parser.add_argument(
        "--quantization_type",
        type=str,
        default="dynamic",
        choices=["dynamic", "float16", "int8"],
        help="Type of quantization to apply"
    )
    
    parser.add_argument(
        "--representative_dataset",
        type=str,
        help="Path to representative dataset for int8 quantization"
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
        "--include_tokenizer",
        action="store_true",
        help="Whether to include the tokenizer in the output directory"
    )
    
    parser.add_argument(
        "--optimize_for",
        type=str,
        default="default",
        choices=["default", "storage", "latency"],
        help="Optimization target for the TFLite model"
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

def convert_to_onnx(model, output_path, batch_size=1, sequence_length=64, spatial_dim=3):
    """
    Convert PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model
        output_path: Path to save the ONNX model
        batch_size: Batch size for the converted model
        sequence_length: Sequence length for the converted model
        spatial_dim: Number of spatial dimensions
        
    Returns:
        onnx_path: Path to the saved ONNX model
    """
    logger.info("Converting PyTorch model to ONNX format")
    
    # Create dummy inputs
    dummy_input_ids = torch.ones((batch_size, sequence_length), dtype=torch.long)
    dummy_attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.long)
    dummy_spatial_coords = torch.zeros((batch_size, spatial_dim), dtype=torch.float)
    
    # Define dynamic axes for inputs
    dynamic_axes = {
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'spatial_coordinates': {0: 'batch_size'}
    }
    
    # Export to ONNX
    onnx_path = os.path.join(output_path, "model.onnx")
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask, None, None, dummy_spatial_coords),
        onnx_path,
        input_names=['input_ids', 'attention_mask', 'spatial_coordinates'],
        output_names=['logits', 'spatial_predictions'],
        dynamic_axes=dynamic_axes,
        opset_version=12,
        do_constant_folding=True,
        verbose=False
    )
    
    logger.info(f"ONNX model saved to {onnx_path}")
    return onnx_path

def create_representative_dataset(data_path, tokenizer, batch_size=1, sequence_length=64):
    """
    Create a representative dataset for int8 quantization.
    
    Args:
        data_path: Path to the dataset
        tokenizer: The tokenizer
        batch_size: Batch size for the converted model
        sequence_length: Sequence length for the converted model
        
    Returns:
        generator: Dataset generator function
    """
    logger.info(f"Loading representative dataset from {data_path}")
    
    # Load dataset
    with open(data_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f.readlines()]
    
    # Tokenize texts
    encodings = []
    for text in texts:
        encoding = tokenizer.encode_plus(
            text,
            max_length=sequence_length,
            padding="max_length",
            truncation=True,
            return_tensors="tf"
        )
        encodings.append((encoding["input_ids"], encoding["attention_mask"]))
    
    def representative_dataset():
        for input_ids, attention_mask in encodings:
            yield [
                input_ids.numpy().astype(np.int32),
                attention_mask.numpy().astype(np.int32)
            ]
    
    return representative_dataset

def convert_onnx_to_tflite(onnx_path, output_path, args):
    """
    Convert ONNX model to TensorFlow Lite format.
    
    Args:
        onnx_path: Path to the ONNX model
        output_path: Path to save the TFLite model
        args: Command line arguments
        
    Returns:
        tflite_path: Path to the saved TFLite model
    """
    logger.info("Converting ONNX model to TensorFlow Lite format")
    
    # Import onnx_tf for ONNX to TensorFlow conversion
    try:
        import onnx
        import onnx_tf
    except ImportError:
        logger.error("Required packages not installed. Please install onnx and onnx-tf.")
        logger.error("pip install onnx onnx-tf")
        sys.exit(1)
    
    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    
    # Convert ONNX to TensorFlow
    tf_rep = onnx_tf.backend.prepare(onnx_model)
    
    # Save TensorFlow model
    tf_model_path = os.path.join(output_path, "tf_model")
    tf_rep.export_graph(tf_model_path)
    
    # Convert TensorFlow to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    
    # Set optimization flags
    if args.optimize_for == "storage":
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    elif args.optimize_for == "latency":
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
    else:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Apply quantization if requested
    if args.quantize:
        if args.quantization_type == "dynamic":
            logger.info("Applying dynamic range quantization")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
        elif args.quantization_type == "float16":
            logger.info("Applying float16 quantization")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
        elif args.quantization_type == "int8":
            logger.info("Applying int8 quantization")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            if args.representative_dataset:
                # Use representative dataset for full integer quantization
                tokenizer = AutoTokenizer.from_pretrained(args.model_path)
                converter.representative_dataset = create_representative_dataset(
                    args.representative_dataset,
                    tokenizer,
                    args.batch_size,
                    args.sequence_length
                )
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
            else:
                logger.warning("Int8 quantization requires a representative dataset. "
                              "Falling back to dynamic range quantization.")
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save TFLite model
    tflite_path = os.path.join(output_path, "spatialLM_model.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    
    logger.info(f"TFLite model saved to {tflite_path}")
    return tflite_path

def save_model_metadata(model, tokenizer, output_path, args):
    """
    Save model metadata for use in Android app.
    
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
        "quantized": args.quantize,
        "quantization_type": args.quantization_type if args.quantize else "none",
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

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load PyTorch model
    model, tokenizer = load_pytorch_model(args.model_path)
    
    # Convert to ONNX
    onnx_path = convert_to_onnx(
        model,
        args.output_dir,
        args.batch_size,
        args.sequence_length,
        args.spatial_dim
    )
    
    # Convert to TFLite
    tflite_path = convert_onnx_to_tflite(onnx_path, args.output_dir, args)
    
    # Save model metadata
    save_model_metadata(model, tokenizer, args.output_dir, args)
    
    # Save tokenizer if requested
    if args.include_tokenizer:
        tokenizer_dir = os.path.join(args.output_dir, "tokenizer")
        tokenizer.save_pretrained(tokenizer_dir)
        logger.info(f"Tokenizer saved to {tokenizer_dir}")
    
    logger.info("Conversion completed successfully!")
    logger.info(f"TFLite model saved to {tflite_path}")

if __name__ == "__main__":
    main()
