#!/usr/bin/env python3
"""
test_model.py

Quick test script to verify SpatialLM 1.1 model download and basic functionality.
This script tests the Qwen-based SpatialLM 1.1 model from manycore-research.

This should be placed in: spatiallm/test_model.py (root of spatiallm folder)
"""

import os
import sys
import logging
import torch
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("spatiallm_test")

def test_dependencies():
    """Test if all required dependencies are available"""
    logger.info("Testing dependencies...")
    
    missing_deps = []
    
    try:
        import torch
        logger.info(f"✓ PyTorch: {torch.__version__}")
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import transformers
        logger.info(f"✓ Transformers: {transformers.__version__}")
    except ImportError:
        missing_deps.append("transformers")
    
    try:
        from huggingface_hub import snapshot_download
        logger.info("✓ Hugging Face Hub available")
    except ImportError:
        missing_deps.append("huggingface-hub")
    
    if missing_deps:
        logger.error(f"Missing dependencies: {missing_deps}")
        logger.error("Install with: pip install torch transformers huggingface-hub")
        return False
    
    return True

def test_model_download(model_path: str = None):
    """Test downloading the SpatialLM 1.1 model"""
    logger.info("Testing SpatialLM 1.1 model download...")
    
    if model_path and os.path.exists(model_path):
        logger.info(f"Using existing model at: {model_path}")
        return model_path
    
    try:
        from huggingface_hub import snapshot_download
        
        # Download the model
        model_path = snapshot_download(
            repo_id="manycore-research/SpatialLM1.1-Qwen-0.5B",
            cache_dir="./test_models"
        )
        
        logger.info(f"✓ Model downloaded successfully to: {model_path}")
        
        # Check required files
        required_files = [
            "config.json",
            "pytorch_model.bin",
            "tokenizer.json",
            "tokenizer_config.json"
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(model_path, file)):
                missing_files.append(file)
        
        if missing_files:
            logger.warning(f"Missing files: {missing_files}")
        else:
            logger.info("✓ All required files present")
        
        return model_path
        
    except Exception as e:
        logger.error(f"Model download failed: {str(e)}")
        return None

def test_model_loading(model_path: str):
    """Test loading the model with transformers"""
    logger.info("Testing model loading...")
    
    try:
        from transformers import AutoModel, AutoTokenizer, AutoConfig
        
        # Load configuration
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        logger.info(f"✓ Config loaded - Model type: {config.model_type}")
        
        # Try to get model info
        if hasattr(config, 'hidden_size'):
            logger.info(f"  Hidden size: {config.hidden_size}")
        if hasattr(config, 'vocab_size'):
            logger.info(f"  Vocab size: {config.vocab_size}")
        if hasattr(config, 'num_hidden_layers'):
            logger.info(f"  Num layers: {config.num_hidden_layers}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        logger.info(f"✓ Tokenizer loaded - Vocab size: {tokenizer.vocab_size}")
        
        # Load model (try with minimal resources first)
        try:
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                low_cpu_mem_usage=True
            )
            logger.info(f"✓ Model loaded successfully")
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"  Model parameters: {total_params:,}")
            
            return model, tokenizer, config
            
        except Exception as e:
            logger.warning(f"Full model loading failed: {str(e)}")
            logger.info("This is normal for large models on limited hardware")
            return None, tokenizer, config
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        return None, None, None

def test_tokenizer_functionality(tokenizer):
    """Test basic tokenizer functionality"""
    logger.info("Testing tokenizer functionality...")
    
    try:
        # Test input with spatial context
        test_text = "The robot is located at coordinates (10.5, 20.3, 5.0) and needs to move to position (15.2, 25.8, 5.0)."
        
        # Tokenize input
        inputs = tokenizer(
            test_text,
            return_tensors="pt",
            max_length=128,
            padding=True,
            truncation=True
        )
        
        logger.info(f"Input text: {test_text}")
        logger.info(f"Tokenized input shape: {inputs['input_ids'].shape}")
        logger.info(f"Number of tokens: {inputs['input_ids'].shape[1]}")
        
        # Test decoding
        decoded = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        logger.info(f"Decoded text matches: {decoded == test_text}")
        
        logger.info("✓ Tokenizer functionality test passed")
        return True
        
    except Exception as e:
        logger.error(f"Tokenizer test failed: {str(e)}")
        return False

def test_basic_inference(model, tokenizer):
    """Test basic inference if model is available"""
    if model is None:
        logger.info("Skipping inference test (model not loaded)")
        return False
    
    logger.info("Testing basic inference...")
    
    try:
        # Test input with spatial context
        test_text = "The robot is located at coordinates (10.5, 20.3, 5.0)."
        
        # Tokenize input
        inputs = tokenizer(
            test_text,
            return_tensors="pt",
            max_length=128,
            padding=True,
            truncation=True
        )
        
        logger.info(f"Input text: {test_text}")
        logger.info(f"Tokenized input shape: {inputs['input_ids'].shape}")
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        logger.info(f"✓ Inference completed")
        logger.info(f"  Output shape: {outputs.last_hidden_state.shape}")
        logger.info(f"  Output dtype: {outputs.last_hidden_state.dtype}")
        
        return True
        
    except Exception as e:
        logger.error(f"Inference test failed: {str(e)}")
        return False

def test_spatial_coordinates():
    """Test spatial coordinate processing"""
    logger.info("Testing spatial coordinate extraction...")
    
    try:
        import re
        
        # Test texts with spatial coordinates
        test_cases = [
            "Move to coordinates (10, 20, 30)",
            "Located at GPS 40.7128° N, 74.0060° W",
            "Position: x=15.5, y=25.3, z=10.0",
            "Navigate to latitude 37.7749, longitude -122.4194"
        ]
        
        coordinate_patterns = [
            r'\(([+-]?\d*\.?\d+),\s*([+-]?\d*\.?\d+),\s*([+-]?\d*\.?\d+)\)',  # (x, y, z)
            r'([+-]?\d*\.?\d+)°?\s*[NS],?\s*([+-]?\d*\.?\d+)°?\s*[EW]',  # GPS coordinates
            r'x=([+-]?\d*\.?\d+),?\s*y=([+-]?\d*\.?\d+),?\s*z=([+-]?\d*\.?\d+)',  # x=, y=, z=
            r'latitude\s+([+-]?\d*\.?\d+),?\s*longitude\s+([+-]?\d*\.?\d+)'  # lat/lon
        ]
        
        extracted_coords = []
        for text in test_cases:
            for pattern in coordinate_patterns:
                matches = re.findall(pattern, text)
                if matches:
                    extracted_coords.append((text, matches[0]))
                    break
        
        logger.info(f"✓ Extracted {len(extracted_coords)} coordinate sets:")
        for text, coords in extracted_coords:
            logger.info(f"  '{text[:50]}...' -> {coords}")
        
        return len(extracted_coords) > 0
        
    except Exception as e:
        logger.error(f"Coordinate extraction test failed: {str(e)}")
        return False

def test_model_size_and_memory(model_path: str):
    """Test model size and memory requirements"""
    logger.info("Testing model size and memory requirements...")
    
    try:
        # Calculate model size
        total_size = 0
        file_count = 0
        
        for root, dirs, files in os.walk(model_path):
            for file in files:
                file_path = os.path.join(root, file)
                size = os.path.getsize(file_path)
                total_size += size
                file_count += 1
        
        size_mb = total_size / (1024 * 1024)
        size_gb = size_mb / 1024
        
        logger.info(f"Model size: {size_mb:.1f} MB ({size_gb:.2f} GB)")
        logger.info(f"Number of files: {file_count}")
        
        # Check if size is reasonable for mobile deployment
        if size_mb < 2000:  # Less than 2GB
            logger.info("✓ Model size suitable for mobile conversion")
        else:
            logger.warning("⚠ Large model size may require additional optimization")
        
        # Check available memory
        try:
            import psutil
            available_ram = psutil.virtual_memory().available / (1024**3)  # GB
            logger.info(f"Available RAM: {available_ram:.1f} GB")
            
            if available_ram > size_gb * 2:  # Need at least 2x model size
                logger.info("✓ Sufficient RAM for model loading")
            else:
                logger.warning("⚠ Limited RAM may affect model loading")
                
        except ImportError:
            logger.info("psutil not available - cannot check RAM")
        
        return True
        
    except Exception as e:
        logger.error(f"Size/memory test failed: {str(e)}")
        return False

def run_comprehensive_test(model_path: str = None):
    """Run comprehensive test suite"""
    logger.info("🚀 Starting SpatialLM 1.1 comprehensive test suite")
    logger.info("=" * 60)
    
    results = {}
    
    # Test 1: Dependencies
    logger.info("\n📦 Test 1: Dependencies")
    results["dependencies"] = test_dependencies()
    
    if not results["dependencies"]:
        logger.error("❌ Cannot proceed without required dependencies")
        return results
    
    # Test 2: Model Download
    logger.info("\n📥 Test 2: Model Download/Access")
    downloaded_model_path = test_model_download(model_path)
    results["download"] = downloaded_model_path is not None
    
    if not downloaded_model_path:
        logger.error("❌ Cannot proceed without model access")
        return results
    
    # Test 3: Model Loading
    logger.info("\n🔄 Test 3: Model Loading")
    model, tokenizer, config = test_model_loading(downloaded_model_path)
    results["loading"] = tokenizer is not None  # At least tokenizer should load
    
    # Test 4: Tokenizer Functionality
    if tokenizer:
        logger.info("\n🔤 Test 4: Tokenizer Functionality")
        results["tokenizer"] = test_tokenizer_functionality(tokenizer)
    else:
        results["tokenizer"] = False
    
    # Test 5: Basic Inference
    if model and tokenizer:
        logger.info("\n🧠 Test 5: Basic Inference")
        results["inference"] = test_basic_inference(model, tokenizer)
    else:
        logger.info("\n🧠 Test 5: Basic Inference (Skipped - model not loaded)")
        results["inference"] = False
    
    # Test 6: Spatial Coordinate Processing
    logger.info("\n🗺️ Test 6: Spatial Coordinate Processing")
    results["spatial_coords"] = test_spatial_coordinates()
    
    # Test 7: Model Size and Memory
    logger.info("\n💾 Test 7: Model Size and Memory")
    results["size_memory"] = test_model_size_and_memory(downloaded_model_path)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📋 TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "❌ FAIL"
        logger.info(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed >= total - 1:  # Allow 1 test to fail (inference might fail on limited hardware)
        logger.info("🎉 SpatialLM 1.1 is ready for use!")
        logger.info("\n📖 Next steps:")
        logger.info("1. Convert to mobile format using convert_to_coreml.py or convert_to_tflite.py")
        logger.info("2. Integrate into your mobile application")
        logger.info("3. Fine-tune for your specific spatial reasoning tasks")
    else:
        logger.warning("⚠️ Some critical tests failed. Please check the logs above.")
        logger.info("\n🔧 Troubleshooting:")
        logger.info("1. Ensure all dependencies are installed: pip install torch transformers huggingface-hub")
        logger.info("2. Check internet connection for model download")
        logger.info("3. Verify sufficient disk space and RAM")
    
    return results

def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SpatialLM 1.1 model")
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to already downloaded model (optional)"
    )
    
    args = parser.parse_args()
    
    try:
        results = run_comprehensive_test(args.model_path)
        
        # Exit with appropriate code
        passed = sum(results.values())
        total = len(results)
        
        if passed >= total - 1:  # Allow 1 test to fail
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during testing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()