# internvl/scripts/convert_to_mobile.py

#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from mobile_converter import MobileConverter

def main():
    parser = argparse.ArgumentParser(description='Convert InternVL model for mobile deployment')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='./models/mobile',
                       help='Output directory for mobile models')
    parser.add_argument('--platform', type=str, nargs='+', choices=['ios', 'android', 'both'],
                       default=['both'], help='Target platform(s)')
    parser.add_argument('--config', type=str, default='config/deployment_config.yaml',
                       help='Path to deployment configuration')
    parser.add_argument('--input_size', type=int, default=224,
                       help='Input image size for mobile optimization')
    parser.add_argument('--max_sequence_length', type=int, default=512,
                       help='Maximum text sequence length')
    parser.add_argument('--validate', action='store_true',
                       help='Validate converted models')
    parser.add_argument('--quantize', action='store_true',
                       help='Apply quantization for smaller model size')
    
    args = parser.parse_args()
    
    # Expand 'both' platform
    platforms = []
    for platform in args.platform:
        if platform == 'both':
            platforms.extend(['ios', 'android'])
        else:
            platforms.append(platform)
    
    # Remove duplicates
    platforms = list(set(platforms))
    
    try:
        print("Initializing mobile converter...")
        
        # Initialize converter
        converter = MobileConverter(args.config)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Input shapes
        input_shape = (1, 3, args.input_size, args.input_size)
        text_input_shape = (1, args.max_sequence_length)
        
        converted_models = {}
        
        # Convert for each platform
        for platform in platforms:
            print(f"\nConverting model for {platform.upper()}...")
            
            platform_dir = output_dir / platform
            platform_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                if platform == 'ios':
                    converted_path = converter.convert_to_coreml(
                        model_path=args.model_path,
                        output_path=str(platform_dir),
                        input_shape=input_shape,
                        text_input_shape=text_input_shape
                    )
                    
                elif platform == 'android':
                    converted_path = converter.convert_to_tflite(
                        model_path=args.model_path,
                        output_path=str(platform_dir),
                        input_shape=input_shape,
                        text_input_shape=text_input_shape
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
                
            except Exception as e:
                print(f"  Error converting for {platform}: {e}")
                continue
        
        # Summary
        print(f"\nConversion Summary:")
        print(f"Original model: {args.model_path}")
        print(f"Output directory: {output_dir}")
        
        for platform, path in converted_models.items():
            print(f"  {platform.upper()}: {path}")
        
        # Create deployment instructions
        instructions_file = output_dir / "deployment_instructions.md"
        create_deployment_instructions(instructions_file, converted_models, args)
        
        print(f"\nDeployment instructions saved to: {instructions_file}")
        print("Mobile conversion completed successfully!")
        
    except Exception as e:
        print(f"Error during mobile conversion: {e}")
        sys.exit(1)

def create_deployment_instructions(output_file: Path, converted_models: dict, args):
    """Create deployment instructions for the converted models."""
    
    instructions = f"""# InternVL Mobile Deployment Instructions

## Converted Models

This directory contains InternVL models converted for mobile deployment:

"""
    
    for platform, model_path in converted_models.items():
        instructions += f"- **{platform.upper()}**: `{model_path}`\n"
    
    instructions += f"""
## Model Configuration

- Input image size: {args.input_size}x{args.input_size}
- Max sequence length: {args.max_sequence_length}
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
   import Vision
   
   // Load the model
   guard let model = try? InternVLMobile(configuration: MLModelConfiguration()) else {{
       fatalError("Failed to load model")
   }}
   
   // Prepare input
   let pixelBuffer = // Your image as CVPixelBuffer
   let textTokens = // Your tokenized text as MLMultiArray
   
   // Make prediction
   let prediction = try model.prediction(image: pixelBuffer, input_ids: textTokens)
   ```

3. **Required Frameworks**:
   - CoreML
   - Vision
   - Accelerate (for preprocessing)

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

3. **Kotlin/Java Code Example**:
   ```kotlin
   import org.tensorflow.lite.Interpreter
   import org.tensorflow.lite.support.image.TensorImage
   
   // Load the model
   val tfliteModel = loadModelFile("model.tflite")
   val interpreter = Interpreter(tfliteModel)
   
   // Prepare inputs
   val imageInput = TensorImage.fromBitmap(bitmap)
   val textInput = // Your tokenized text as IntArray
   
   // Run inference
   val outputs = arrayOf(FloatArray(outputSize))
   interpreter.run(arrayOf(imageInput.buffer, textInput), outputs)
   ```

## Performance Optimization

### iOS:
- Use Neural Engine when available
- Enable compute precision optimization
- Consider model quantization for smaller size

### Android:
- Use GPU delegate for faster inference
- Enable NNAPI delegate when supported
- Apply dynamic quantization

## Integration Notes

1. **Preprocessing**: Ensure input images are properly normalized and resized
2. **Tokenization**: Use the same tokenizer as during training
3. **Postprocessing**: Apply appropriate output formatting for your use case
4. **Error Handling**: Implement proper error handling for model loading and inference

## Troubleshooting

### Common Issues:

1. **Model loading fails**:
   - Check file path and permissions
   - Verify model format compatibility

2. **Inference errors**:
   - Validate input shapes and types
   - Ensure proper preprocessing

3. **Performance issues**:
   - Enable hardware acceleration
   - Consider model optimization

### Support:
- Check model validation results in conversion logs
- Test with sample inputs before production deployment
"""
    
    with open(output_file, 'w') as f:
        f.write(instructions)

if __name__ == "__main__":
    main()

---

# internvl/scripts/train_model.py

#!/usr/bin/env python3

import argparse
import sys
import torch
from pathlib import Path

# Add src to path  
sys.path.append(str(Path(__file__).parent.parent / "src"))

from trainer import InternVLTrainer
from data_processor import DataProcessor
from model_downloader import ModelDownloader

def main():
    parser = argparse.ArgumentParser(description='Train InternVL model from scratch')
    parser.add_argument('--model_name', type=str, default='internvl2-2b',
                       help='Model name to download and train')
    parser.add_argument('--train_data', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--val_data', type=str,
                       help='Path to validation data')
    parser.add_argument('--output_dir', type=str, default='./models/trained',
                       help='Output directory for trained model')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Path to training configuration')
    parser.add_argument('--download_model', action='store_true',
                       help='Download model before training')
    parser.add_argument('--resume_from_checkpoint', type=str,
                       help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    try:
        model_path = None
        
        # Download model if requested
        if args.download_model:
            print("Downloading model...")
            downloader = ModelDownloader()
            model_path, _ = downloader.download_model(args.model_name)
            print(f"Model downloaded to: {model_path}")
        else:
            # Assume model is already available
            model_path = f"./models/pretrained/{args.model_name}"
            if not Path(model_path).exists():
                print(f"Model not found at {model_path}. Use --download_model to download it first.")
                sys.exit(1)
        
        print("Initializing trainer...")
        
        # Initialize trainer
        trainer = InternVLTrainer(
            model_path=model_path,
            config_path=args.config
        )
        
        # Setup model and tokenizer
        trainer.setup_model_and_tokenizer()
        
        print("Setting up data loaders...")
        
        # Initialize data processor
        data_processor = DataProcessor(
            trainer.tokenizer,
            trainer.config['training']['data']
        )
        
        # Create data loaders
        dataloaders = data_processor.create_dataloaders(
            train_path=args.train_data,
            val_path=args.val_data,
            batch_size=trainer.config['training']['batch_size']
        )
        
        print("Starting training...")
        
        # Start training
        trainer.train(
            train_dataloader=dataloaders['train'],
            val_dataloader=dataloaders.get('val'),
            output_dir=args.output_dir,
            resume_from_checkpoint=args.resume_from_checkpoint
        )
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
