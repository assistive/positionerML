#!/usr/bin/env python3
"""
SpatialLM 1.1 Test Script

Quick test script to verify SpatialLM 1.1 model download and basic functionality.
This script tests the Qwen-based SpatialLM 1.1 model from manycore-research.
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

def test_model_download():
    """Test downloading the SpatialLM 1.1 model"""
    logger.info("Testing SpatialLM 1.1 model download...")
    
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
        config = AutoConfig.from_pretrained(model_path)
        logger.info(f"✓ Config loaded - Model type: {config.model_type}")
        logger.info(f"  Hidden size: {getattr(config, 'hidden_size', 'N/A')}")
        logger.info(f"  Vocab size: {getattr(config, 'vocab_size', 'N/A')}")
        logger.info(f"  Num layers: {getattr(config, 'num_hidden_layers', 'N/A')}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        logger.info(f"✓ Tokenizer loaded - Vocab size: {tokenizer.vocab_size}")
        
        # Load model
        model = AutoModel.from_pretrained(
            model_path, 
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model, tokenizer, config
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        return None, None, None

def test_basic_inference(model, tokenizer):
    """Test basic inference with spatial context"""
    logger.info("Testing basic inference...")
    
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

def test_model_conversion_readiness(model_path: str):
    """Test readiness for mobile conversion"""
    logger.info("Testing mobile conversion readiness...")
    
    try:
        # Check model size
        model_files = list(Path(model_path).glob("*.bin"))
        if model_files:
            total_size_mb = sum(f.stat().st_size for f in model_files) / (1024 * 1024)
            logger.info(f"Model size: {total_size_mb:.1f} MB")
            
            if total_size_mb < 2000:  # Less than 2GB
                logger.info("✓ Model size suitable for mobile conversion")
            else:
                logger.warning("⚠ Large model size may require additional optimization")
        
        # Check for quantization support
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            model_type = config.get("model_type", "unknown")
            if model_type in ["qwen", "qwen2"]:
                logger.info("✓ Qwen architecture supports quantization")
            else:
                logger.info(f"Model type: {model_type}")
        
        # Check for mobile config files
        mobile_configs = ["mobile_config.json", "optimization_config.json"]
        for config_file in mobile_configs:
            if os.path.exists(os.path.join(model_path, config_file)):
                logger.info(f"✓ Found {config_file}")
            else:
                logger.info(f"- {config_file} not found (will be created during conversion)")
        
        return True
        
    except Exception as e:
        logger.error(f"Conversion readiness test failed: {str(e)}")
        return False

def run_comprehensive_test():
    """Run comprehensive test suite"""
    logger.info("🚀 Starting SpatialLM 1.1 comprehensive test suite")
    logger.info("=" * 60)
    
    results = {}
    
    # Test 1: Model Download
    logger.info("\n📥 Test 1: Model Download")
    model_path = test_model_download()
    results["download"] = model_path is not None
    
    if not model_path:
        logger.error("❌ Cannot proceed without model download")
        return results
    
    # Test 2: Model Loading
    logger.info("\n🔄 Test 2: Model Loading")
    model, tokenizer, config = test_model_loading(model_path)
    results["loading"] = all([model, tokenizer, config])
    
    # Test 3: Basic Inference
    if model and tokenizer:
        logger.info("\n🧠 Test 3: Basic Inference")
        results["inference"] = test_basic_inference(model, tokenizer)
    else:
        results["inference"] = False
    
    # Test 4: Spatial Coordinate Processing
    logger.info("\n🗺️ Test 4: Spatial Coordinate Processing")
    results["spatial_coords"] = test_spatial_coordinates()
    
    # Test 5: Mobile Conversion Readiness
    logger.info("\n📱 Test 5: Mobile Conversion Readiness")
    results["conversion_ready"] = test_model_conversion_readiness(model_path)
    
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
    
    if passed == total:
        logger.info("🎉 All tests passed! SpatialLM 1.1 is ready for use.")
        logger.info("\n📖 Next steps:")
        logger.info("1. Fine-tune the model for your specific task")
        logger.info("2. Convert to mobile format (CoreML/TensorFlow Lite)")
        logger.info("3. Deploy to your application")
    else:
        logger.warning("⚠️ Some tests failed. Please check the logs above.")
    
    return results

def main():
    """Main test function"""
    try:
        results = run_comprehensive_test()
        
        # Exit with appropriate code
        if all(results.values()):
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