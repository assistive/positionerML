# bert_mobile/scripts/convert_to_mobile.py

#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from mobile_converter import BERTMobileConverter

def main():
    parser = argparse.ArgumentParser(description='Convert BERT model for mobile deployment')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained BERT model')
    parser.add_argument('--output_dir', type=str, default='./models/mobile',
                       help='Output directory for mobile models')
    parser.add_argument('--platforms', type=str, nargs='+', 
                       choices=['ios', 'android', 'both'], default=['both'],
                       help='Target platforms')
    parser.add_argument('--sequence_length', type=int, default=128,
                       help='Maximum sequence length for mobile')
    parser.add_argument('--task_type', type=str, choices=['classification', 'feature_extraction'],
                       default='classification', help='Task type')
    parser.add_argument('--quantize', action='store_true', default=True,
                       help='Apply quantization')
    parser.add_argument('--optimize', action='store_true',
                       help='Apply mobile optimizations')
    parser.add_argument('--validate', action='store_true',
                       help='Validate converted models')
    parser.add_argument('--config', type=str, default='config/mobile_config.yaml',
                       help='Path to mobile configuration')
    parser.add_argument('--benchmark', action='store_true',
                       help='Benchmark converted models')
    
    args = parser.parse_args()
    
    # Expand 'both' platform
    platforms = []
    for platform in args.platforms:
        if platform == 'both':
            platforms.extend(['ios', 'android'])
        else:
            platforms.append(platform)
    
    # Remove duplicates
    platforms = list(set(platforms))
    
    try:
        print("Initializing mobile converter...")
        
        # Initialize converter
        converter = BERTMobileConverter(args.config)
        
        # Apply optimizations if requested
        model_path = args.model_path
        if args.optimize:
            print("Applying mobile optimizations...")
            model_path = converter.optimize_for_mobile(
                args.model_path, 
                converter.config['mobile']['optimization']
            )
        
        # Convert for each platform
        converted_models = {}
        
        for platform in platforms:
            print(f"\nConverting model for {platform.upper()}...")
            
            platform_dir = Path(args.output_dir) / platform
            platform_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                if platform == 'ios':
                    converted_path = converter.convert_to_coreml(
                        model_path=model_path,
                        output_path=str(platform_dir),
                        sequence_length=args.sequence_length,
                        task_type=args.task_type
                    )
                    
                elif platform == 'android':
                    converted_path = converter.convert_to_tflite(
                        model_path=model_path,
                        output_path=str(platform_dir),
                        sequence_length=args.sequence_length,
                        task_type=args.task_type
                    )
                
                converted_models[platform] = converted_path
                
                # Get model size info
                size_info = converter.get_model_size(converted_path)
                print(f"  Converted model size: {size_info['size_mb']:.2f} MB")
                
                # Validate conversion if requested
                if args.validate:
                    print(f"  Validating {platform} conversion...")
                    validation_results = converter.validate_conversion(
                        original_model_path=args.model_path,
                        converted_model_path=converted_path,
                        platform=platform
                    )
                    
                    if validation_results['conversion_successful']:
                        print(f"  ✓ Validation passed (MAE: {validation_results['mae']:.6f})")
                    else:
                        print(f"  ✗ Validation failed: {validation_results.get('error', 'Unknown error')}")
                
                # Benchmark performance if requested
                if args.benchmark:
                    print(f"  Benchmarking {platform} model...")
                    benchmark_results = converter.benchmark_model(converted_path, platform)
                    if benchmark_results:
                        print(f"  Average inference time: {benchmark_results['avg_inference_time_ms']:.1f}ms")
                        print(f"  Throughput: {benchmark_results['fps']:.2f} FPS")
                
            except Exception as e:
                print(f"  Error converting for {platform}: {e}")
                continue
        
        # Create deployment instructions
        create_deployment_instructions(Path(args.output_dir), converted_models, args)
        
        # Summary
        print(f"\nConversion Summary:")
        print(f"Original model: {args.model_path}")
        print(f"Output directory: {args.output_dir}")
        
        for platform, path in converted_models.items():
            print(f"  {platform.upper()}: {path}")
        
        print("\nMobile conversion completed successfully!")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

def create_deployment_instructions(output_dir: Path, converted_models: dict, args):
    """Create deployment instructions for the converted models."""
    
    instructions = f"""# BERT Mobile Deployment Instructions

This directory contains BERT models converted for mobile deployment.

## Converted Models

"""
    
    for platform, model_path in converted_models.items():
        instructions += f"- **{platform.upper()}**: `{model_path}`\n"
    
    instructions += f"""
## Model Configuration

- Task type: {args.task_type}
- Max sequence length: {args.sequence_length}
- Quantization: {'Enabled' if args.quantize else 'Disabled'}
- Original model: {args.model_path}

## iOS Deployment (CoreML)

### Integration Steps:

1. **Add Model to Xcode Project**:
   ```
   Drag the .mlmodel file into your Xcode project
   ```

2. **Swift Code Example**:
   ```swift
   import CoreML
   
   // Load the model
   guard let model = try? BERTMobile(configuration: MLModelConfiguration()) else {{
       fatalError("Failed to load model")
   }}
   
   // Prepare input
   let inputIds = // Your tokenized text as MLMultiArray
   let attentionMask = // Your attention mask as MLMultiArray
   
   // Make prediction
   let prediction = try model.prediction(input_ids: inputIds, attention_mask: attentionMask)
   ```

3. **Required Frameworks**:
   - CoreML
   - NaturalLanguage (for tokenization)

## Android Deployment (TensorFlow Lite)

### Integration Steps:

1. **Add to Android Project**:
   ```
   Place the .tflite file in app/src/main/assets/
   ```

2. **Add Dependencies** (app/build.gradle):
   ```gradle
   implementation 'org.tensorflow:tensorflow-lite:2.13.0'
   implementation 'org.tensorflow:tensorflow-lite-gpu:2.13.0'
   implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
   ```

3. **Kotlin Code Example**:
   ```kotlin
   import org.tensorflow.lite.Interpreter
   
   // Load the model
   val interpreter = Interpreter(loadModelFile("bert_mobile.tflite"))
   
   // Prepare inputs
   val inputIds = // Your tokenized text as IntArray
   val attentionMask = // Your attention mask as IntArray
   
   // Run inference
   val outputs = Array(1) {{ FloatArray(numClasses) }}
   interpreter.run(arrayOf(inputIds, attentionMask), outputs)
   ```

## Performance Optimization

### iOS:
- Use Neural Engine when available
- Enable float16 precision
- Batch processing for multiple texts

### Android:
- Use GPU delegate for faster inference
- Enable NNAPI delegate when supported
- Consider dynamic quantization

## Integration Notes

1. **Tokenization**: Use the same tokenizer as during training
2. **Preprocessing**: Ensure proper text normalization and padding
3. **Postprocessing**: Apply softmax for classification probabilities
4. **Error Handling**: Implement proper error handling for model loading

## Performance Expectations

- **iOS**: ~50-200ms inference time on modern devices
- **Android**: ~100-300ms inference time with GPU acceleration
- **Memory**: ~50-100MB model size after quantization

## Troubleshooting

### Common Issues:

1. **Model loading fails**:
   - Check file path and permissions
   - Verify model format compatibility

2. **Inference errors**:
   - Validate input shapes and types
   - Ensure proper tokenization

3. **Performance issues**:
   - Enable hardware acceleration
   - Consider model optimization

### Support:
- Check tokenizer compatibility
- Validate input preprocessing
- Test with sample inputs before production
"""
    
    with open(output_dir / "DEPLOYMENT_GUIDE.md", 'w') as f:
        f.write(instructions)
    
    print(f"Deployment guide created: {output_dir / 'DEPLOYMENT_GUIDE.md'}")

if __name__ == "__main__":
    main()

