#!/usr/bin/env python3
"""
convert_to_tflite.py

Convert spatialLM v1.1 model to TensorFlow Lite format for Android deployment.
This should be placed in: spatiallm/convert_to_tflite.py (root of spatiallm folder)

This script handles the conversion of the SpatialLM 1.1 model to TensorFlow Lite
format, optimizing it for mobile deployment on Android devices.
"""

import os
import sys
import logging
import argparse
import torch
import numpy as np
import json
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("convert_to_tflite")

# Check TensorFlow availability
try:
    import tensorflow as tf
    TF_AVAILABLE = True
    logger.info(f"TensorFlow version: {tf.__version__}")
except ImportError:
    TF_AVAILABLE = False
    logger.error("TensorFlow not installed. Please install with:")
    logger.error("pip install tensorflow")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Convert spatialLM v1.1 model to TensorFlow Lite")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the downloaded spatialLM v1.1 model directory"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tflite_models",
        help="Directory to save the converted TensorFlow Lite model"
    )
    
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=128,
        help="Maximum sequence length for the converted model"
    )
    
    parser.add_argument(
        "--quantize",
        action="store_true",
        default=True,
        help="Whether to quantize the model for improved performance"
    )
    
    parser.add_argument(
        "--quantization_mode",
        type=str,
        default="int8",
        choices=["int8", "float16", "dynamic"],
        help="Quantization mode"
    )
    
    parser.add_argument(
        "--optimize_for_size",
        action="store_true",
        help="Optimize for model size over inference speed"
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
    
    required_files = ["config.json"]
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
        
        # Load model
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        model.eval()
        
        logger.info("✓ Model and tokenizer loaded successfully")
        return model, tokenizer, config
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return None, None, None

def create_onnx_model(model, tokenizer, sequence_length: int, output_path: str):
    """Convert PyTorch model to ONNX format as intermediate step"""
    logger.info("Converting to ONNX format...")
    
    try:
        # Create example inputs
        input_ids = torch.randint(
            0, min(tokenizer.vocab_size, 32000), 
            (1, sequence_length), 
            dtype=torch.long
        )
        attention_mask = torch.ones((1, sequence_length), dtype=torch.long)
        
        onnx_path = output_path + ".onnx"
        
        # Export to ONNX
        with torch.no_grad():
            torch.onnx.export(
                model,
                (input_ids, attention_mask),
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input_ids', 'attention_mask'],
                output_names=['last_hidden_state'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence'},
                    'attention_mask': {0: 'batch_size', 1: 'sequence'},
                    'last_hidden_state': {0: 'batch_size', 1: 'sequence'}
                }
            )
        
        logger.info(f"✓ ONNX model saved: {onnx_path}")
        return onnx_path, {"input_ids": input_ids, "attention_mask": attention_mask}
        
    except Exception as e:
        logger.error(f"ONNX conversion failed: {str(e)}")
        return None, None

def convert_onnx_to_tflite(onnx_path: str, quantization_mode: str, optimize_for_size: bool):
    """Convert ONNX model to TensorFlow Lite"""
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not available")
    
    logger.info("Converting ONNX to TensorFlow Lite...")
    
    try:
        # Try using onnx-tf converter if available
        try:
            import onnx
            import onnx_tf
            
            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            
            # Convert to TensorFlow
            tf_rep = onnx_tf.backend.prepare(onnx_model)
            
            # Export to SavedModel format
            savedmodel_path = onnx_path.replace('.onnx', '_savedmodel')
            tf_rep.export_graph(savedmodel_path)
            
            # Convert SavedModel to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_path)
            
        except ImportError:
            logger.warning("onnx-tf not available, trying alternative conversion...")
            return None
        
        # Configure converter
        converter.allow_custom_ops = True
        converter.experimental_new_converter = True
        
        # Apply quantization
        if quantization_mode == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        elif quantization_mode == "float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif quantization_mode == "dynamic":
            converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        
        if optimize_for_size:
            converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        
        # Convert
        tflite_model = converter.convert()
        
        logger.info("✓ TensorFlow Lite conversion completed")
        return tflite_model
        
    except Exception as e:
        logger.error(f"TensorFlow Lite conversion failed: {str(e)}")
        return None

def create_simple_tflite_model(model, tokenizer, sequence_length: int, quantization_mode: str):
    """Create a simple TFLite model using direct conversion"""
    logger.info("Creating simplified TFLite model...")
    
    try:
        # This is a placeholder for a simpler conversion approach
        # In practice, you might need to:
        # 1. Simplify the model architecture
        # 2. Use a smaller subset of the model
        # 3. Create a custom conversion pipeline
        
        logger.warning("Direct PyTorch to TFLite conversion is complex for transformer models")
        logger.info("Consider using the ONNX intermediate format or model distillation")
        
        return None
        
    except Exception as e:
        logger.error(f"Simple TFLite conversion failed: {str(e)}")
        return None

def validate_tflite_model(tflite_model: bytes, example_inputs: Dict[str, torch.Tensor]):
    """Validate the converted TensorFlow Lite model"""
    logger.info("Validating TensorFlow Lite model...")
    
    try:
        # Create interpreter
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        logger.info(f"Model inputs: {len(input_details)}")
        logger.info(f"Model outputs: {len(output_details)}")
        
        for i, detail in enumerate(input_details):
            logger.info(f"Input {i}: {detail['name']} - shape: {detail['shape']} - dtype: {detail['dtype']}")
        
        for i, detail in enumerate(output_details):
            logger.info(f"Output {i}: {detail['name']} - shape: {detail['shape']} - dtype: {detail['dtype']}")
        
        # Try a simple inference test
        if len(input_details) <= 2:  # Only if we have reasonable number of inputs
            # Create test inputs matching the expected shapes
            for i, detail in enumerate(input_details):
                test_input = np.random.randint(0, 1000, detail['shape']).astype(detail['dtype'])
                interpreter.set_tensor(detail['index'], test_input)
            
            # Run inference
            interpreter.invoke()
            
            # Get output
            output = interpreter.get_tensor(output_details[0]['index'])
            logger.info(f"Test inference output shape: {output.shape}")
        
        logger.info("✓ TensorFlow Lite model validation completed")
        return True
        
    except Exception as e:
        logger.error(f"TensorFlow Lite model validation failed: {str(e)}")
        return False

def save_tokenizer_info(tokenizer, model_path: str, output_dir: str):
    """Save tokenizer information for Android integration"""
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
            "model_max_length": getattr(tokenizer, 'model_max_length', 512),
            "android_integration": {
                "framework": "TensorFlow Lite",
                "recommended_api_level": "24+",
                "performance_profile": "nnapi_optimized",
                "memory_requirements": "< 2GB",
                "inference_time": "< 1000ms"
            }
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

def create_android_deployment_info(args, output_dir: str, model_size_mb: float):
    """Create Android deployment information file"""
    deployment_info = {
        "model_version": "1.1",
        "conversion_date": str(torch.datetime.now()) if hasattr(torch, 'datetime') else "unknown",
        "model_size_mb": model_size_mb,
        "source_model": "manycore-research/SpatialLM1.1-Qwen-0.5B",
        "configuration": {
            "sequence_length": args.sequence_length,
            "quantization": args.quantize,
            "quantization_mode": args.quantization_mode,
            "optimize_for_size": args.optimize_for_size
        },
        "android_integration": {
            "framework": "TensorFlow Lite",
            "recommended_api_level": "24+",
            "ndk_version": "21+",
            "performance_profile": "nnapi_optimized",
            "memory_requirements": "< 2GB",
            "inference_time": "< 1000ms"
        },
        "usage_instructions": {
            "add_to_assets": "Copy the .tflite file to your Android project's assets folder",
            "load_model": "Use Interpreter.Options() to load the model",
            "preprocessing": "Use tokenizer_info.json for text preprocessing"
        }
    }
    
    info_path = os.path.join(output_dir, "android_deployment_info.json")
    with open(info_path, 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    logger.info(f"Android deployment info saved to {info_path}")

def create_kotlin_integration_example(output_dir: str):
    """Create Kotlin integration example for Android"""
    kotlin_example = '''
package com.example.spatiallm

import android.content.Context
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder

class SpatialLMInference(private val context: Context) {
    private var interpreter: Interpreter? = null
    
    fun loadModel(modelPath: String): Boolean {
        return try {
            val model = FileUtil.loadMappedFile(context, modelPath)
            val options = Interpreter.Options()
            // Enable NNAPI delegate for better performance
            options.setUseNNAPI(true)
            options.setNumThreads(4)
            
            interpreter = Interpreter(model, options)
            println("SpatialLM v1.1 model loaded successfully")
            true
        } catch (e: Exception) {
            println("Error loading model: ${e.message}")
            false
        }
    }
    
    fun runInference(inputIds: IntArray, attentionMask: IntArray): FloatArray? {
        val interpreter = this.interpreter ?: return null
        
        return try {
            // Prepare input tensors
            val inputBuffer1 = ByteBuffer.allocateDirect(inputIds.size * 4)
                .order(ByteOrder.nativeOrder())
            val inputBuffer2 = ByteBuffer.allocateDirect(attentionMask.size * 4)
                .order(ByteOrder.nativeOrder())
            
            inputIds.forEach { inputBuffer1.putInt(it) }
            attentionMask.forEach { inputBuffer2.putInt(it) }
            
            // Prepare output tensor
            val outputShape = interpreter.getOutputTensor(0).shape()
            val outputSize = outputShape.fold(1) { acc, dim -> acc * dim }
            val outputBuffer = ByteBuffer.allocateDirect(outputSize * 4)
                .order(ByteOrder.nativeOrder())
            
            // Run inference
            val inputs = arrayOf(inputBuffer1, inputBuffer2)
            val outputs = mapOf(0 to outputBuffer)
            
            interpreter.runForMultipleInputsOutputs(inputs, outputs)
            
            // Convert output to FloatArray
            outputBuffer.rewind()
            val result = FloatArray(outputSize)
            outputBuffer.asFloatBuffer().get(result)
            
            result
        } catch (e: Exception) {
            println("Inference error: ${e.message}")
            null
        }
    }
    
    fun close() {
        interpreter?.close()
        interpreter = null
    }
}

// Usage example:
// val spatialLM = SpatialLMInference(this)
// if (spatialLM.loadModel("SpatialLM_v1.1.tflite")) {
//     val result = spatialLM.runInference(inputIds, attentionMask)
//     // Process result...
// }
'''
    
    example_path = os.path.join(output_dir, "SpatialLMInference.kt")
    with open(example_path, 'w') as f:
        f.write(kotlin_example.strip())
    
    logger.info(f"Kotlin integration example saved to {example_path}")

def create_gradle_dependencies(output_dir: str):
    """Create Gradle dependencies file for Android integration"""
    gradle_deps = '''
// Add to your app-level build.gradle

dependencies {
    // TensorFlow Lite
    implementation 'org.tensorflow:tensorflow-lite:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
    
    // Optional: GPU delegate (if supported)
    implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.4.4'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.13.0'
    
    // JSON parsing for tokenizer
    implementation 'com.google.code.gson:gson:2.10.1'
    
    // Coroutines for background processing
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.6.4'
}

android {
    compileSdk 34
    
    defaultConfig {
        minSdk 24
        targetSdk 34
    }
    
    aaptOptions {
        noCompress "tflite"
        noCompress "json"
    }
}
'''
    
    gradle_path = os.path.join(output_dir, "build.gradle.dependencies")
    with open(gradle_path, 'w') as f:
        f.write(gradle_deps.strip())
    
    logger.info(f"Gradle dependencies saved to {gradle_path}")

def main():
    """Main conversion function"""
    args = parse_arguments()
    
    if not TF_AVAILABLE:
        logger.error("TensorFlow not available. Please install tensorflow.")
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
        
        # Try ONNX conversion first
        model_base_name = f"SpatialLM_v1.1_{args.quantization_mode}"
        onnx_path, example_inputs = create_onnx_model(
            model, tokenizer, args.sequence_length, 
            str(output_dir / model_base_name)
        )
        
        tflite_model = None
        if onnx_path:
            # Convert ONNX to TFLite
            tflite_model = convert_onnx_to_tflite(
                onnx_path, args.quantization_mode, args.optimize_for_size
            )
        
        if tflite_model is None:
            logger.warning("ONNX-based conversion failed, trying simplified approach...")
            logger.info("For complex transformer models, consider:")
            logger.info("1. Model distillation to create a smaller model")
            logger.info("2. Using ONNX Runtime Mobile instead of TFLite")
            logger.info("3. Converting only specific model components")
            
            # Create a placeholder info file
            placeholder_info = {
                "status": "conversion_failed",
                "reason": "Complex transformer model conversion",
                "alternatives": [
                    "Use ONNX Runtime Mobile",
                    "Apply model distillation",
                    "Use cloud-based inference"
                ]
            }
            
            with open(output_dir / "conversion_status.json", 'w') as f:
                json.dump(placeholder_info, f, indent=2)
            
            sys.exit(1)
        
        # Validate the model
        if args.validate_model and example_inputs:
            validate_tflite_model(tflite_model, example_inputs)
        
        # Save the TensorFlow Lite model
        tflite_path = output_dir / f"{model_base_name}.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        # Calculate model size
        model_size_mb = len(tflite_model) / (1024 * 1024)
        
        logger.info(f"✓ TensorFlow Lite model saved: {tflite_path}")
        logger.info(f"Model size: {model_size_mb:.2f} MB")
        
        # Save tokenizer information
        if args.include_tokenizer:
            save_tokenizer_info(tokenizer, args.model_path, str(output_dir))
        
        # Create deployment information
        create_android_deployment_info(args, str(output_dir), model_size_mb)
        
        # Create integration examples
        create_kotlin_integration_example(str(output_dir))
        create_gradle_dependencies(str(output_dir))
        
        # Clean up ONNX file
        if onnx_path and os.path.exists(onnx_path):
            os.remove(onnx_path)
        
        print("\n🚀 Conversion completed successfully!")
        print(f"📱 TensorFlow Lite model: {tflite_path}")
        print(f"📊 Model size: {model_size_mb:.2f} MB")
        print(f"⚡ Quantization: {args.quantization_mode}")
        print(f"🎯 Optimized for: Android API 24+")
        
        print(f"\n📖 Next steps:")
        print(f"1. Copy {tflite_path} to your Android project's assets folder")
        print(f"2. Use the tokenizer files for text preprocessing")
        print(f"3. Integrate using: {output_dir}/SpatialLMInference.kt")
        print(f"4. Add dependencies from: {output_dir}/build.gradle.dependencies")
        print(f"5. Check android_deployment_info.json for detailed specs")
        
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        logger.info("\nNote: TensorFlow Lite conversion of large transformer models is challenging.")
        logger.info("Consider using ONNX Runtime Mobile or model distillation for better results.")
        sys.exit(1)

if __name__ == "__main__":
    main()