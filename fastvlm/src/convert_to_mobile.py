#!/usr/bin/env python3
# fastvlm/scripts/convert_to_mobile.py

import argparse
import os
import sys
from pathlib import Path
import yaml
import logging
import json
from typing import List, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.mobile_converter import MobileConverter, MobileOptimizationConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Convert FastVLM models for mobile deployment")
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained FastVLM model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/mobile",
        help="Output directory for converted models"
    )
    
    # Platform arguments
    parser.add_argument(
        "--platform",
        type=str,
        nargs="+",
        choices=["ios", "android", "onnx", "all"],
        default=["all"],
        help="Target platforms for conversion"
    )
    
    # Optimization arguments
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Apply optimizations before conversion"
    )
    parser.add_argument(
        "--target_size_mb",
        type=float,
        default=100.0,
        help="Target model size in MB"
    )
    parser.add_argument(
        "--quantization_bits",
        type=int,
        choices=[4, 8, 16],
        default=8,
        help="Quantization bits"
    )
    parser.add_argument(
        "--pruning_sparsity",
        type=float,
        default=0.5,
        help="Pruning sparsity ratio (0.0 to 1.0)"
    )
    
    # Model configuration
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for converted model"
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=77,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Input image size"
    )
    parser.add_argument(
        "--num_vision_tokens",
        type=int,
        default=49,
        help="Number of vision tokens (for mobile optimization)"
    )
    
    # Other arguments
    parser.add_argument(
        "--config",
        type=str,
        help="Path to conversion configuration file"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark converted models"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify converted models"
    )
    
    args = parser.parse_args()
    
    # Expand platform list
    platforms = []
    for platform in args.platform:
        if platform == "all":
            platforms.extend(["ios", "android", "onnx"])
        else:
            platforms.append(platform)
    platforms = list(set(platforms))  # Remove duplicates
    
    # Create optimization config
    optimization_config = MobileOptimizationConfig(
        target_size_mb=args.target_size_mb,
        quantization_bits=args.quantization_bits,
        pruning_sparsity=args.pruning_sparsity,
        use_dynamic_quantization=True,
        optimize_for_inference=True,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        image_size=args.image_size,
        num_vision_tokens=args.num_vision_tokens
    )
    
    # Create converter
    converter = MobileConverter(config_path=args.config)
    
    # Log configuration
    logger.info("Conversion Configuration:")
    logger.info(f"  Model path: {args.model_path}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Platforms: {platforms}")
    logger.info(f"  Optimization: {args.optimize}")
    logger.info(f"  Target size: {args.target_size_mb} MB")
    logger.info(f"  Quantization: {args.quantization_bits} bits")
    logger.info(f"  Image size: {args.image_size}")
    logger.info(f"  Sequence length: {args.sequence_length}")
    
    # Apply optimization if requested
    model_path = args.model_path
    if args.optimize:
        logger.info("Applying optimizations...")
        model_path = converter.optimize_model(model_path, optimization_config)
        logger.info(f"Optimized model saved to: {model_path}")
    
    # Convert for each platform
    converted_models = {}
    
    for platform in platforms:
        logger.info(f"\nConverting for {platform}...")
        
        try:
            output_path = Path(args.output_dir) / platform
            
            if platform == "ios":
                converted_path = converter.convert_to_coreml(
                    model_path,
                    str(output_path),
                    optimization_config
                )
            elif platform == "android":
                converted_path = converter.convert_to_tflite(
                    model_path,
                    str(output_path),
                    optimization_config
                )
            elif platform == "onnx":
                converted_path = converter.convert_to_onnx(
                    model_path,
                    str(output_path),
                    optimization_config
                )
            
            converted_models[platform] = converted_path
            logger.info(f"✓ {platform} model saved to: {converted_path}")
            
            # Get model size
            model_size = Path(converted_path).stat().st_size / (1024 * 1024)
            logger.info(f"  Model size: {model_size:.2f} MB")
            
        except Exception as e:
            logger.error(f"✗ Failed to convert for {platform}: {e}")
            continue
    
    # Benchmark if requested
    if args.benchmark and converted_models:
        logger.info("\nBenchmarking converted models...")
        
        benchmark_results = {}
        for platform, model_path in converted_models.items():
            logger.info(f"Benchmarking {platform} model...")
            
            try:
                results = converter.benchmark_mobile_model(
                    model_path,
                    platform,
                    num_runs=100
                )
                benchmark_results[platform] = results
                
                logger.info(f"  Average latency: {results['avg_latency_ms']:.2f} ms")
                logger.info(f"  P90 latency: {results.get('p90_latency_ms', 'N/A')}")
                logger.info(f"  P99 latency: {results.get('p99_latency_ms', 'N/A')}")
                
            except Exception as e:
                logger.error(f"Benchmarking failed for {platform}: {e}")
    
    # Save conversion summary
    summary_path = Path(args.output_dir) / "conversion_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "source_model": args.model_path,
        "optimization_applied": args.optimize,
        "optimization_config": optimization_config.__dict__,
        "converted_models": converted_models,
        "platforms": platforms,
        "benchmark_results": benchmark_results if args.benchmark else None
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nConversion summary saved to: {summary_path}")
    
    # Create deployment guide
    guide_path = Path(args.output_dir) / "deployment_guide.md"
    create_deployment_guide(guide_path, converted_models, optimization_config)
    logger.info(f"Deployment guide saved to: {guide_path}")
    
    logger.info("\nConversion completed successfully!")
    logger.info(f"All converted models saved to: {args.output_dir}")


def create_deployment_guide(output_path: Path, converted_models: dict, config: MobileOptimizationConfig):
    """Create a deployment guide for the converted models."""
    
    guide_content = f"""# FastVLM Mobile Deployment Guide

## Converted Models

This directory contains FastVLM models optimized and converted for mobile deployment.

### Model Information
- Input image size: {config.image_size}x{config.image_size}
- Maximum sequence length: {config.sequence_length}
- Number of vision tokens: {config.num_vision_tokens}
- Quantization: {config.quantization_bits}-bit
- Batch size: {config.batch_size}

### Available Models
"""
    
    for platform, model_path in converted_models.items():
        model_size = Path(model_path).stat().st_size / (1024 * 1024)
        guide_content += f"\n- **{platform.upper()}**: `{Path(model_path).name}` ({model_size:.1f} MB)"
    
    guide_content += """

## iOS Deployment

### Requirements
- iOS 15.0+
- Xcode 13.0+
- CoreML framework

### Integration Steps

1. Add the `.mlpackage` to your Xcode project
2. Import the generated Swift code:
   ```swift
   import FastVLM
   ```

3. Initialize and use the model:
   ```swift
   let fastvlm = try FastVLM()
   let result = try await fastvlm.predict(image: uiImage, text: "What is in this image?")
   ```

### Performance Tips
- Use `.all` compute units for best performance
- Enable Neural Engine when available
- Consider batching requests for efficiency

## Android Deployment

### Requirements
- Android API 24+
- TensorFlow Lite 2.13.0+

### Integration Steps

1. Add the `.tflite` file to `app/src/main/assets/`
2. Add dependencies to `build.gradle`:
   ```gradle
   implementation 'org.tensorflow:tensorflow-lite:2.13.0'
   implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
   ```

3. Use the provided Kotlin code:
   ```kotlin
   val fastvlm = FastVLM(context)
   val result = fastvlm.predict(bitmap, "What is in this image?")
   ```

### Performance Tips
- Enable GPU delegate for faster inference
- Use NNAPI delegate when available
- Consider using background threads

## ONNX Deployment

The ONNX model can be used with:
- ONNX Runtime (CPU, GPU, Mobile)
- Web deployment with ONNX.js
- Edge devices with ONNX Runtime Edge

### Example Usage
```python
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
outputs = session.run(None, {
    "pixel_values": image_array,
    "input_ids": token_ids,
    "attention_mask": attention_mask
})
```

## Optimization Details

The models have been optimized with:
- {config.quantization_bits}-bit quantization
- Pruning with {config.pruning_sparsity*100:.0f}% sparsity
- Architecture optimizations for mobile inference
- Reduced vision tokens from 256 to {config.num_vision_tokens}

## Troubleshooting

### iOS Issues
- Ensure minimum deployment target is iOS 15.0
- Check that compute units are properly configured
- Verify input shapes match expected dimensions

### Android Issues
- Ensure TFLite delegates are properly initialized
- Check memory allocation for large models
- Verify input preprocessing matches training

### General Tips
- Always normalize images to [-1, 1] or [0, 1] as expected
- Ensure text tokenization matches the training tokenizer
- Monitor memory usage on device
- Test on various devices for compatibility

## Support

For additional help:
- Check the example applications in the repository
- Review the API documentation
- File issues on GitHub
"""
    
    with open(output_path, 'w') as f:
        f.write(guide_content)


if __name__ == "__main__":
    main()
