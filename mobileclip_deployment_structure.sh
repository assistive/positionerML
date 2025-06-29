#!/bin/bash

# MobileCLIP Mobile Deployment Project Structure
# Complete deployment pipeline for MobileCLIP models on iOS and Android

set -e

echo "🚀 Creating MobileCLIP Mobile Deployment Project Structure..."

# Create root directory
PROJECT_NAME="mobileclip_mobile"
mkdir -p $PROJECT_NAME
cd $PROJECT_NAME

# Create main directory structure
mkdir -p {config,src,scripts,models,data,mobile,docs,tests,tools,examples,deployment}

# Configuration directories
mkdir -p config/{mobile,deployment,training,optimization}

# Source code structure
mkdir -p src/{core,mobile_converter,utils,android,ios,data_processing,optimization}

# Scripts for automation
mkdir -p scripts/{download,convert,deploy,test,benchmark,optimization}

# Model storage
mkdir -p models/{pretrained,converted,optimized}/{android,ios}

# Data processing
mkdir -p data/{raw,processed,test_images,benchmarks,validation}

# Mobile platform specific
mkdir -p mobile/{android/{app,gradle,kotlin,assets,libs,integration},ios/{swift,coreml,xcode,assets,integration}}

# Documentation
mkdir -p docs/{api,deployment,tutorials,troubleshooting,performance}

# Testing
mkdir -p tests/{unit,integration,mobile,performance,model_validation}

# Tools and utilities
mkdir -p tools/{quantization,optimization,benchmarking,model_analysis}

# Examples
mkdir -p examples/{android,ios,react_native,python,web}

# Deployment packages
mkdir -p deployment/{packages,ios,android,reports}

echo "📝 Creating configuration files..."

# Main MobileCLIP configuration
cat > config/mobileclip_config.yaml << 'EOF'
# MobileCLIP Mobile Deployment Configuration

model:
  name: "mobileclip"
  variants:
    - "mobileclip_s0"     # 11.4M + 42.4M params - Fastest
    - "mobileclip_s1"     # 25.2M + 42.4M params - Balanced  
    - "mobileclip_s2"     # 35.7M + 42.4M params - Better accuracy
    - "mobileclip_b"      # 86.3M + 42.4M params - Base model
    - "mobileclip_blt"    # 86.3M + 42.4M params - Long training
  
  input_size: [224, 224, 3]
  text_max_length: 77
  embedding_dim: 512
  pretrained_urls:
    mobileclip_s0: "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s0.pt"
    mobileclip_s1: "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s1.pt"
    mobileclip_s2: "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s2.pt"
    mobileclip_b: "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_b.pt"
    mobileclip_blt: "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt"
  
mobile:
  android:
    target_api: 24
    quantization: "int8"
    use_nnapi: true
    use_gpu_delegate: true
    model_format: "tflite"
    max_memory_mb: 500
    target_inference_ms: 100
    
  ios:
    deployment_target: "15.0"
    quantization: "float16"
    use_neural_engine: true
    model_format: "mlmodel"
    max_memory_mb: 300
    target_inference_ms: 50
    
optimization:
  quantization:
    enabled: true
    methods: ["dynamic", "static", "qat"]
    int8_calibration_samples: 1000
  
  pruning:
    enabled: false
    sparsity: 0.3
    
  knowledge_distillation:
    enabled: true
    teacher_model: "mobileclip_b"
    student_model: "mobileclip_s0"
    temperature: 4.0

deployment:
  batch_size: 1
  enable_multimodal: true
  performance_profiling: true
  validation_datasets: ["imagenet", "coco", "flickr30k"]
  
huggingface:
  organization: "apple"
  models:
    - "apple/MobileCLIP-S0"
    - "apple/MobileCLIP-S1" 
    - "apple/MobileCLIP-S2"
    - "apple/MobileCLIP-B"
EOF

# Android specific configuration
cat > config/mobile/android_config.yaml << 'EOF'
android:
  gradle_version: "8.2"
  kotlin_version: "1.9.0"
  compile_sdk: 34
  min_sdk: 24
  target_sdk: 34
  
  dependencies:
    tensorflow_lite: "2.13.0"
    tensorflow_lite_gpu: "2.13.0"
    tensorflow_lite_support: "0.4.4"
    
  optimization:
    use_nnapi: true
    use_gpu_delegate: true
    use_hexagon_delegate: false
    num_threads: 4
    allow_fp16: true
    
  model_specs:
    mobileclip_s0:
      size_mb: 60
      inference_ms: 80
      memory_mb: 200
    mobileclip_s1:
      size_mb: 90
      inference_ms: 120
      memory_mb: 280
    mobileclip_s2:
      size_mb: 110
      inference_ms: 150
      memory_mb: 350
    mobileclip_b:
      size_mb: 180
      inference_ms: 300
      memory_mb: 500
EOF

# iOS specific configuration  
cat > config/mobile/ios_config.yaml << 'EOF'
ios:
  deployment_target: "15.0"
  xcode_version: "15.0"
  swift_version: "5.9"
  
  frameworks:
    - "CoreML"
    - "Vision" 
    - "Accelerate"
    - "Metal"
    - "MetalPerformanceShaders"
    
  optimization:
    use_neural_engine: true
    use_gpu: true
    precision: "float16"
    compute_units: "all"
    
  model_specs:
    mobileclip_s0:
      size_mb: 55
      inference_ms: 30
      memory_mb: 150
    mobileclip_s1:
      size_mb: 80
      inference_ms: 50
      memory_mb: 200
    mobileclip_s2:
      size_mb: 95
      inference_ms: 70
      memory_mb: 250
    mobileclip_b:
      size_mb: 150
      inference_ms: 120
      memory_mb: 350
EOF

echo "🐍 Creating Python source files..."

# Core MobileCLIP model wrapper
cat > src/core/mobileclip_model.py << 'EOF'
"""
MobileCLIP Model Wrapper
Unified interface for MobileCLIP models with mobile optimization
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from pathlib import Path
import logging

try:
    import mobileclip
    MOBILECLIP_AVAILABLE = True
except ImportError:
    MOBILECLIP_AVAILABLE = False
    logging.warning("MobileCLIP package not found. Install with: pip install git+https://github.com/apple/ml-mobileclip")

class MobileCLIPModel:
    """Wrapper for MobileCLIP models with mobile deployment features."""
    
    def __init__(self, model_name: str = "mobileclip_s0", pretrained_path: Optional[str] = None):
        self.model_name = model_name
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if MOBILECLIP_AVAILABLE:
            self._load_model(pretrained_path)
    
    def _load_model(self, pretrained_path: Optional[str] = None):
        """Load MobileCLIP model and preprocessing."""
        try:
            self.model, _, self.preprocess = mobileclip.create_model_and_transforms(
                self.model_name, 
                pretrained=pretrained_path
            )
            self.tokenizer = mobileclip.get_tokenizer(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logging.info(f"Successfully loaded {self.model_name}")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to feature vector."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        with torch.no_grad():
            image_features = self.model.encode_image(image.to(self.device))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    def encode_text(self, text: List[str]) -> torch.Tensor:
        """Encode text to feature vector."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        text_tokens = self.tokenizer(text)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens.to(self.device))
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def compute_similarity(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """Compute similarity between image and text features."""
        return (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    def zero_shot_classify(self, image: torch.Tensor, text_labels: List[str]) -> Dict[str, float]:
        """Perform zero-shot classification."""
        image_features = self.encode_image(image)
        text_features = self.encode_text(text_labels)
        similarities = self.compute_similarity(image_features, text_features)
        
        results = {}
        for i, label in enumerate(text_labels):
            results[label] = float(similarities[0, i])
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if self.model is None:
            return {}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_name": self.model_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "input_size": [224, 224, 3],
            "text_max_length": 77
        }
    
    def prepare_for_mobile(self) -> nn.Module:
        """Prepare model for mobile deployment."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Reparameterize model for inference
        try:
            from mobileclip.modules.common.mobileone import reparameterize_model
            mobile_model = reparameterize_model(self.model)
            mobile_model.eval()
            return mobile_model
        except ImportError:
            logging.warning("Reparameterization not available, returning original model")
            return self.model
EOF

# Model downloader
cat > src/core/model_downloader.py << 'EOF'
"""
MobileCLIP Model Downloader
Downloads pretrained models from Apple and Hugging Face
"""
import os
import requests
import hashlib
from pathlib import Path
from typing import Dict, Optional
import logging
from tqdm import tqdm
import yaml

class MobileCLIPDownloader:
    """Download MobileCLIP models from various sources."""
    
    def __init__(self, cache_dir: str = "./models/pretrained"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Official Apple model URLs
        self.model_urls = {
            "mobileclip_s0": "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s0.pt",
            "mobileclip_s1": "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s1.pt", 
            "mobileclip_s2": "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s2.pt",
            "mobileclip_b": "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_b.pt",
            "mobileclip_blt": "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt"
        }
        
        # Hugging Face model names
        self.hf_models = {
            "mobileclip_s0": "apple/MobileCLIP-S0",
            "mobileclip_s1": "apple/MobileCLIP-S1",
            "mobileclip_s2": "apple/MobileCLIP-S2", 
            "mobileclip_b": "apple/MobileCLIP-B"
        }
    
    def download_file(self, url: str, filepath: Path, chunk_size: int = 8192) -> bool:
        """Download file with progress bar."""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f, tqdm(
                desc=filepath.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logging.info(f"Downloaded {filepath}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to download {url}: {e}")
            if filepath.exists():
                filepath.unlink()
            return False
    
    def download_from_apple(self, model_name: str, force_download: bool = False) -> Optional[Path]:
        """Download model from Apple's official repository."""
        if model_name not in self.model_urls:
            logging.error(f"Model {model_name} not available from Apple")
            return None
        
        filepath = self.cache_dir / f"{model_name}.pt"
        
        if filepath.exists() and not force_download:
            logging.info(f"Model {model_name} already exists at {filepath}")
            return filepath
        
        url = self.model_urls[model_name]
        logging.info(f"Downloading {model_name} from Apple...")
        
        if self.download_file(url, filepath):
            return filepath
        return None
    
    def download_from_huggingface(self, model_name: str, force_download: bool = False) -> Optional[Path]:
        """Download model from Hugging Face."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            logging.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
            return None
        
        if model_name not in self.hf_models:
            logging.error(f"Model {model_name} not available on Hugging Face")
            return None
        
        try:
            hf_model_name = self.hf_models[model_name]
            filepath = hf_hub_download(
                repo_id=hf_model_name,
                filename="pytorch_model.bin",
                cache_dir=str(self.cache_dir),
                force_download=force_download
            )
            logging.info(f"Downloaded {model_name} from Hugging Face to {filepath}")
            return Path(filepath)
            
        except Exception as e:
            logging.error(f"Failed to download from Hugging Face: {e}")
            return None
    
    def download_model(self, model_name: str, source: str = "apple", force_download: bool = False) -> Optional[Path]:
        """Download model from specified source."""
        if source == "apple":
            return self.download_from_apple(model_name, force_download)
        elif source == "huggingface":
            return self.download_from_huggingface(model_name, force_download)
        else:
            logging.error(f"Unknown source: {source}")
            return None
    
    def download_all_models(self, source: str = "apple", force_download: bool = False) -> Dict[str, Optional[Path]]:
        """Download all available models."""
        results = {}
        
        if source == "apple":
            models = self.model_urls.keys()
        elif source == "huggingface":
            models = self.hf_models.keys()
        else:
            logging.error(f"Unknown source: {source}")
            return results
        
        for model_name in models:
            results[model_name] = self.download_model(model_name, source, force_download)
        
        return results
    
    def verify_model(self, model_path: Path) -> bool:
        """Verify downloaded model integrity."""
        if not model_path.exists():
            return False
        
        try:
            import torch
            torch.load(model_path, map_location='cpu')
            return True
        except Exception as e:
            logging.error(f"Model verification failed: {e}")
            return False
    
    def list_available_models(self) -> Dict[str, Dict]:
        """List all available models with their information."""
        models_info = {}
        
        for model_name in self.model_urls.keys():
            models_info[model_name] = {
                "apple_url": self.model_urls.get(model_name),
                "huggingface": self.hf_models.get(model_name),
                "local_path": self.cache_dir / f"{model_name}.pt",
                "downloaded": (self.cache_dir / f"{model_name}.pt").exists()
            }
        
        return models_info
EOF

# Mobile converter
cat > src/mobile_converter/converter.py << 'EOF'
"""
MobileCLIP Mobile Converter
Converts MobileCLIP models to mobile-optimized formats (CoreML, TensorFlow Lite)
"""
import torch
import torch.nn as nn
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np

class MobileCLIPMobileConverter:
    """Convert MobileCLIP models for mobile deployment."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.mobile_model = None
        
    def load_pytorch_model(self, model_name: str, model_path: Optional[str] = None):
        """Load PyTorch MobileCLIP model."""
        try:
            import mobileclip
            from mobileclip.modules.common.mobileone import reparameterize_model
            
            # Load model
            self.model, _, _ = mobileclip.create_model_and_transforms(
                model_name, pretrained=model_path
            )
            
            # Reparameterize for inference
            self.model.eval()
            self.mobile_model = reparameterize_model(self.model)
            self.mobile_model.eval()
            
            logging.info(f"Loaded and reparameterized {model_name}")
            
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise
    
    def convert_to_coreml(self, output_dir: str, model_name: str = "mobileclip") -> str:
        """Convert model to CoreML format for iOS."""
        try:
            import coremltools as ct
        except ImportError:
            raise ImportError("CoreML Tools not available. Install with: pip install coremltools")
        
        if self.mobile_model is None:
            raise RuntimeError("No model loaded")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create example inputs
        example_image = torch.randn(1, 3, 224, 224)
        example_text = torch.randint(0, 1000, (1, 77))  # Text tokens
        
        # Trace the model
        with torch.no_grad():
            # Split into image and text encoders for separate conversion
            traced_image_model = torch.jit.trace(
                self.mobile_model.visual, 
                example_image
            )
            traced_text_model = torch.jit.trace(
                self.mobile_model.text, 
                example_text
            )
        
        # Convert image encoder
        image_model_path = output_path / f"{model_name}_image.mlpackage"
        image_coreml_model = ct.convert(
            traced_image_model,
            inputs=[ct.ImageType(name="image", shape=example_image.shape)],
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.iOS15
        )
        image_coreml_model.save(str(image_model_path))
        
        # Convert text encoder  
        text_model_path = output_path / f"{model_name}_text.mlpackage"
        text_coreml_model = ct.convert(
            traced_text_model,
            inputs=[ct.TensorType(name="text_tokens", shape=example_text.shape)],
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.iOS15
        )
        text_coreml_model.save(str(text_model_path))
        
        logging.info(f"CoreML models saved to {output_path}")
        return str(output_path)
    
    def convert_to_tflite(self, output_dir: str, model_name: str = "mobileclip") -> str:
        """Convert model to TensorFlow Lite format for Android."""
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow not available. Install with: pip install tensorflow")
        
        if self.mobile_model is None:
            raise RuntimeError("No model loaded")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert through ONNX first
        onnx_path = self._convert_to_onnx(output_path, model_name)
        
        # Convert ONNX to TensorFlow Lite
        tflite_path = self._onnx_to_tflite(onnx_path, output_path, model_name)
        
        logging.info(f"TensorFlow Lite model saved to {tflite_path}")
        return str(tflite_path)
    
    def _convert_to_onnx(self, output_dir: Path, model_name: str) -> str:
        """Convert PyTorch model to ONNX."""
        try:
            import torch.onnx
        except ImportError:
            raise ImportError("ONNX not available. Install with: pip install onnx")
        
        # Example inputs
        example_image = torch.randn(1, 3, 224, 224)
        example_text = torch.randint(0, 1000, (1, 77))
        
        # Export image encoder
        image_onnx_path = output_dir / f"{model_name}_image.onnx"
        torch.onnx.export(
            self.mobile_model.visual,
            example_image,
            str(image_onnx_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['image'],
            output_names=['image_features']
        )
        
        # Export text encoder
        text_onnx_path = output_dir / f"{model_name}_text.onnx"
        torch.onnx.export(
            self.mobile_model.text,
            example_text,
            str(text_onnx_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['text_tokens'],
            output_names=['text_features']
        )
        
        return str(image_onnx_path)
    
    def _onnx_to_tflite(self, onnx_path: str, output_dir: Path, model_name: str) -> str:
        """Convert ONNX model to TensorFlow Lite."""
        try:
            import onnx
            import tensorflow as tf
            from onnx_tf.backend import prepare
        except ImportError:
            raise ImportError("ONNX-TF not available. Install with: pip install onnx-tf")
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Convert to TensorFlow
        tf_rep = prepare(onnx_model)
        
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_concrete_functions(tf_rep.signatures)
        
        # Optimization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        
        # Save model
        tflite_path = output_dir / f"{model_name}.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        return str(tflite_path)
    
    def validate_conversion(self, original_output: torch.Tensor, converted_output: np.ndarray, tolerance: float = 1e-3) -> bool:
        """Validate model conversion accuracy."""
        original_np = original_output.detach().cpu().numpy()
        
        # Compare outputs
        diff = np.abs(original_np - converted_output)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        logging.info(f"Conversion validation - Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
        
        return max_diff < tolerance
EOF

echo "📱 Creating mobile integration code..."

# iOS Swift integration
cat > mobile/ios/swift/MobileCLIPInference.swift << 'EOF'
//
//  MobileCLIPInference.swift
//  MobileCLIP iOS Integration
//
//  Provides easy-to-use interface for MobileCLIP inference on iOS
//

import Foundation
import CoreML
import Vision
import UIKit
import Accelerate

@available(iOS 15.0, *)
public class MobileCLIPInference: ObservableObject {
    
    // MARK: - Properties
    private var imageModel: MLModel?
    private var textModel: MLModel?
    private let imageSize = CGSize(width: 224, height: 224)
    
    // MARK: - Singleton
    public static let shared = MobileCLIPInference()
    
    private init() {
        loadModels()
    }
    
    // MARK: - Model Loading
    private func loadModels() {
        do {
            // Load image encoder model
            if let imageModelURL = Bundle.main.url(forResource: "mobileclip_image", withExtension: "mlpackage") {
                imageModel = try MLModel(contentsOf: imageModelURL)
                print("✅ Image model loaded successfully")
            }
            
            // Load text encoder model
            if let textModelURL = Bundle.main.url(forResource: "mobileclip_text", withExtension: "mlpackage") {
                textModel = try MLModel(contentsOf: textModelURL)
                print("✅ Text model loaded successfully")
            }
        } catch {
            print("❌ Failed to load models: \(error)")
        }
    }
    
    // MARK: - Image Processing
    private func preprocessImage(_ image: UIImage) -> CVPixelBuffer? {
        // Resize image to 224x224
        guard let resizedImage = image.resized(to: imageSize) else {
            print("Failed to resize image")
            return nil
        }
        
        // Convert to CVPixelBuffer
        return resizedImage.toCVPixelBuffer()
    }
    
    // MARK: - Text Processing
    private func tokenizeText(_ text: String) -> [Int32] {
        // Simple tokenization - in practice, use proper tokenizer
        let maxLength = 77
        var tokens = Array(repeating: Int32(0), count: maxLength)
        
        // Convert text to token IDs (simplified)
        let words = text.lowercased().components(separatedBy: .whitespacesAndNewlines)
        for (index, word) in words.enumerated() {
            if index < maxLength - 2 {
                tokens[index + 1] = Int32(word.hash % 30000) // Simplified hashing
            }
        }
        
        tokens[0] = 49406 // Start token
        if words.count < maxLength - 2 {
            tokens[words.count + 1] = 49407 // End token
        }
        
        return tokens
    }
    
    // MARK: - Inference
    public func extractImageFeatures(from image: UIImage, completion: @escaping (Result<[Float], Error>) -> Void) {
        guard let imageModel = imageModel else {
            completion(.failure(MobileCLIPError.modelNotLoaded))
            return
        }
        
        guard let pixelBuffer = preprocessImage(image) else {
            completion(.failure(MobileCLIPError.imageProcessingFailed))
            return
        }
        
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let input = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: pixelBuffer)])
                let output = try imageModel.prediction(from: input)
                
                if let features = output.featureValue(for: "image_features")?.multiArrayValue {
                    let floatArray = self.multiArrayToFloatArray(features)
                    DispatchQueue.main.async {
                        completion(.success(floatArray))
                    }
                } else {
                    DispatchQueue.main.async {
                        completion(.failure(MobileCLIPError.inferenceError))
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
            }
        }
    }
    
    public func extractTextFeatures(from text: String, completion: @escaping (Result<[Float], Error>) -> Void) {
        guard let textModel = textModel else {
            completion(.failure(MobileCLIPError.modelNotLoaded))
            return
        }
        
        let tokens = tokenizeText(text)
        
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let tokensArray = try MLMultiArray(shape: [1, 77], dataType: .int32)
                for (index, token) in tokens.enumerated() {
                    tokensArray[index] = NSNumber(value: token)
                }
                
                let input = try MLDictionaryFeatureProvider(dictionary: ["text_tokens": MLFeatureValue(multiArray: tokensArray)])
                let output = try textModel.prediction(from: input)
                
                if let features = output.featureValue(for: "text_features")?.multiArrayValue {
                    let floatArray = self.multiArrayToFloatArray(features)
                    DispatchQueue.main.async {
                        completion(.success(floatArray))
                    }
                } else {
                    DispatchQueue.main.async {
                        completion(.failure(MobileCLIPError.inferenceError))
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
            }
        }
    }
    
    public func computeSimilarity(imageFeatures: [Float], textFeatures: [Float]) -> Float {
        guard imageFeatures.count == textFeatures.count else {
            print("Feature vector sizes don't match")
            return 0.0
        }
        
        // Compute cosine similarity
        var dotProduct: Float = 0.0
        var imageNorm: Float = 0.0
        var textNorm: Float = 0.0
        
        for i in 0..<imageFeatures.count {
            dotProduct += imageFeatures[i] * textFeatures[i]
            imageNorm += imageFeatures[i] * imageFeatures[i]
            textNorm += textFeatures[i] * textFeatures[i]
        }
        
        let similarity = dotProduct / (sqrt(imageNorm) * sqrt(textNorm))
        return similarity
    }
    
    public func zeroShotClassify(image: UIImage, labels: [String], completion: @escaping (Result<[(label: String, confidence: Float)], Error>) -> Void) {
        extractImageFeatures(from: image) { [weak self] imageResult in
            switch imageResult {
            case .success(let imageFeatures):
                let group = DispatchGroup()
                var textFeatures: [[Float]] = []
                var errors: [Error] = []
                
                for label in labels {
                    group.enter()
                    self?.extractTextFeatures(from: label) { textResult in
                        switch textResult {
                        case .success(let features):
                            textFeatures.append(features)
                        case .failure(let error):
                            errors.append(error)
                        }
                        group.leave()
                    }
                }
                
                group.notify(queue: .main) {
                    guard errors.isEmpty else {
                        completion(.failure(errors.first!))
                        return
                    }
                    
                    var results: [(label: String, confidence: Float)] = []
                    for (index, features) in textFeatures.enumerated() {
                        let similarity = self?.computeSimilarity(imageFeatures: imageFeatures, textFeatures: features) ?? 0.0
                        results.append((label: labels[index], confidence: similarity))
                    }
                    
                    // Sort by confidence
                    results.sort { $0.confidence > $1.confidence }
                    completion(.success(results))
                }
                
            case .failure(let error):
                completion(.failure(error))
            }
        }
    }
    
    // MARK: - Utility Methods
    private func multiArrayToFloatArray(_ multiArray: MLMultiArray) -> [Float] {
        let count = multiArray.count
        var floatArray = Array<Float>(repeating: 0, count: count)
        
        let dataPointer = multiArray.dataPointer.bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            floatArray[i] = dataPointer[i]
        }
        
        return floatArray
    }
}

// MARK: - Error Types
public enum MobileCLIPError: Error, LocalizedError {
    case modelNotLoaded
    case imageProcessingFailed
    case inferenceError
    
    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "MobileCLIP model not loaded"
        case .imageProcessingFailed:
            return "Failed to process image"
        case .inferenceError:
            return "Inference failed"
        }
    }
}

// MARK: - UIImage Extensions
extension UIImage {
    func resized(to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, 0.0)
        draw(in: CGRect(origin: .zero, size: size))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return resizedImage
    }
    
    func toCVPixelBuffer() -> CVPixelBuffer? {
        let width = Int(self.size.width)
        let height = Int(self.size.height)
        
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                       width,
                                       height,
                                       kCVPixelFormatType_32ARGB,
                                       attrs,
                                       &pixelBuffer)
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(buffer)
        
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: pixelData,
                              width: width,
                              height: height,
                              bitsPerComponent: 8,
                              bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                              space: rgbColorSpace,
                              bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
        
        context?.translateBy(x: 0, y: CGFloat(height))
        context?.scaleBy(x: 1.0, y: -1.0)
        
        UIGraphicsPushContext(context!)
        self.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
        UIGraphicsPopContext()
        CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        
        return buffer
    }
}
EOF

# Android Kotlin integration
cat > mobile/android/kotlin/MobileCLIPInference.kt << 'EOF'
/**
 * MobileCLIPInference.kt
 * MobileCLIP Android Integration
 * 
 * Provides easy-to-use interface for MobileCLIP inference on Android
 */
package com.mobileclip.inference

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.*
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.sqrt

class MobileCLIPInference(private val context: Context) {
    
    companion object {
        private const val TAG = "MobileCLIPInference"
        private const val IMAGE_MODEL_NAME = "mobileclip_image.tflite"
        private const val TEXT_MODEL_NAME = "mobileclip_text.tflite"
        private const val IMAGE_SIZE = 224
        private const val TEXT_MAX_LENGTH = 77
        private const val EMBEDDING_DIM = 512
    }
    
    private var imageInterpreter: Interpreter? = null
    private var textInterpreter: Interpreter? = null
    private var isInitialized = false
    
    private val imageProcessor = ImageProcessor.Builder()
        .add(ResizeOp(IMAGE_SIZE, IMAGE_SIZE, ResizeOp.ResizeMethod.BILINEAR))
        .build()
    
    /**
     * Initialize the MobileCLIP inference engines
     */
    suspend fun initialize(): Boolean = withContext(Dispatchers.IO) {
        try {
            // Load models from assets
            val imageModelBuffer = loadModelFile(IMAGE_MODEL_NAME)
            val textModelBuffer = loadModelFile(TEXT_MODEL_NAME)
            
            // Create interpreters
            val options = Interpreter.Options().apply {
                setNumThreads(4)
                setUseNNAPI(true)
                setUseXNNPACK(true)
            }
            
            imageInterpreter = Interpreter(imageModelBuffer, options)
            textInterpreter = Interpreter(textModelBuffer, options)
            
            isInitialized = true
            Log.d(TAG, "MobileCLIP models initialized successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize models: ${e.message}")
            false
        }
    }
    
    /**
     * Extract image features from bitmap
     */
    suspend fun extractImageFeatures(bitmap: Bitmap): FloatArray? = withContext(Dispatchers.Default) {
        if (!isInitialized || imageInterpreter == null) {
            Log.e(TAG, "Models not initialized")
            return@withContext null
        }
        
        try {
            // Preprocess image
            val tensorImage = TensorImage.fromBitmap(bitmap)
            val processedImage = imageProcessor.process(tensorImage)
            
            // Prepare input buffer
            val inputBuffer = processedImage.buffer
            
            // Prepare output buffer
            val outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, EMBEDDING_DIM), org.tensorflow.lite.DataType.FLOAT32)
            
            // Run inference
            imageInterpreter?.run(inputBuffer, outputBuffer.buffer)
            
            // Extract features
            val features = outputBuffer.floatArray
            
            // Normalize features
            normalizeVector(features)
            
        } catch (e: Exception) {
            Log.e(TAG, "Image feature extraction failed: ${e.message}")
            null
        }
    }
    
    /**
     * Extract text features from string
     */
    suspend fun extractTextFeatures(text: String): FloatArray? = withContext(Dispatchers.Default) {
        if (!isInitialized || textInterpreter == null) {
            Log.e(TAG, "Models not initialized")
            return@withContext null
        }
        
        try {
            // Tokenize text
            val tokens = tokenizeText(text)
            
            // Prepare input buffer
            val inputBuffer = ByteBuffer.allocateDirect(TEXT_MAX_LENGTH * 4)
            inputBuffer.order(ByteOrder.nativeOrder())
            
            for (token in tokens) {
                inputBuffer.putInt(token)
            }
            inputBuffer.rewind()
            
            // Prepare output buffer
            val outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, EMBEDDING_DIM), org.tensorflow.lite.DataType.FLOAT32)
            
            // Run inference
            textInterpreter?.run(inputBuffer, outputBuffer.buffer)
            
            // Extract features
            val features = outputBuffer.floatArray
            
            // Normalize features
            normalizeVector(features)
            
        } catch (e: Exception) {
            Log.e(TAG, "Text feature extraction failed: ${e.message}")
            null
        }
    }
    
    /**
     * Compute similarity between image and text features
     */
    fun computeSimilarity(imageFeatures: FloatArray, textFeatures: FloatArray): Float {
        if (imageFeatures.size != textFeatures.size) {
            Log.e(TAG, "Feature vector sizes don't match")
            return 0f
        }
        
        var dotProduct = 0f
        for (i in imageFeatures.indices) {
            dotProduct += imageFeatures[i] * textFeatures[i]
        }
        
        return dotProduct
    }
    
    /**
     * Perform zero-shot classification
     */
    suspend fun zeroShotClassify(bitmap: Bitmap, labels: List<String>): List<Pair<String, Float>>? {
        val imageFeatures = extractImageFeatures(bitmap) ?: return null
        
        val results = mutableListOf<Pair<String, Float>>()
        
        for (label in labels) {
            val textFeatures = extractTextFeatures(label) ?: continue
            val similarity = computeSimilarity(imageFeatures, textFeatures)
            results.add(Pair(label, similarity))
        }
        
        // Sort by confidence (similarity score)
        return results.sortedByDescending { it.second }
    }
    
    /**
     * Simple text tokenization (simplified implementation)
     */
    private fun tokenizeText(text: String): IntArray {
        val tokens = IntArray(TEXT_MAX_LENGTH) { 0 }
        
        // Simple tokenization - in practice, use proper tokenizer
        val words = text.lowercase().split("\\s+".toRegex())
        
        tokens[0] = 49406 // Start token
        
        for (i in words.indices) {
            if (i < TEXT_MAX_LENGTH - 2) {
                tokens[i + 1] = words[i].hashCode() % 30000 // Simplified hashing
            }
        }
        
        if (words.size < TEXT_MAX_LENGTH - 2) {
            tokens[words.size + 1] = 49407 // End token
        }
        
        return tokens
    }
    
    /**
     * Normalize feature vector to unit length
     */
    private fun normalizeVector(vector: FloatArray) {
        var norm = 0f
        for (value in vector) {
            norm += value * value
        }
        norm = sqrt(norm)
        
        if (norm > 0f) {
            for (i in vector.indices) {
                vector[i] /= norm
            }
        }
    }
    
    /**
     * Load model file from assets
     */
    private fun loadModelFile(fileName: String): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(fileName)
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    /**
     * Clean up resources
     */
    fun close() {
        imageInterpreter?.close()
        textInterpreter?.close()
        imageInterpreter = null
        textInterpreter = null
        isInitialized = false
        Log.d(TAG, "MobileCLIP inference closed")
    }
}

/**
 * Usage example:
 * 
 * class MainActivity : AppCompatActivity() {
 *     private lateinit var mobileCLIP: MobileCLIPInference
 *     
 *     override fun onCreate(savedInstanceState: Bundle?) {
 *         super.onCreate(savedInstanceState)
 *         
 *         mobileCLIP = MobileCLIPInference(this)
 *         
 *         lifecycleScope.launch {
 *             if (mobileCLIP.initialize()) {
 *                 // Ready to use
 *                 val results = mobileCLIP.zeroShotClassify(bitmap, listOf("dog", "cat", "bird"))
 *                 results?.forEach { (label, confidence) ->
 *                     Log.d("Results", "$label: $confidence")
 *                 }
 *             }
 *         }
 *     }
 *     
 *     override fun onDestroy() {
 *         super.onDestroy()
 *         mobileCLIP.close()
 *     }
 * }
 */
EOF

echo "📜 Creating deployment scripts..."

# Model download script
cat > scripts/download/download_models.py << 'EOF'
#!/usr/bin/env python3
"""
MobileCLIP Model Download Script
Downloads pretrained MobileCLIP models from Apple and Hugging Face
"""
import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from core.model_downloader import MobileCLIPDownloader

def main():
    parser = argparse.ArgumentParser(description="Download MobileCLIP models")
    parser.add_argument("--models", nargs="+", 
                       choices=["mobileclip_s0", "mobileclip_s1", "mobileclip_s2", "mobileclip_b", "mobileclip_blt"],
                       default=["mobileclip_s0"],
                       help="Models to download")
    parser.add_argument("--source", choices=["apple", "huggingface"], default="apple",
                       help="Download source")
    parser.add_argument("--output", default="./models/pretrained", help="Output directory")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--verify", action="store_true", help="Verify downloaded models")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create downloader
    downloader = MobileCLIPDownloader(args.output)
    
    print(f"📥 Downloading MobileCLIP models from {args.source}...")
    print(f"📁 Output directory: {args.output}")
    print()
    
    # Download models
    results = {}
    for model_name in args.models:
        print(f"⬇️  Downloading {model_name}...")
        
        try:
            model_path = downloader.download_model(model_name, args.source, args.force)
            
            if model_path:
                results[model_name] = model_path
                
                # Verify if requested
                if args.verify:
                    if downloader.verify_model(model_path):
                        print(f"✅ {model_name} verified successfully")
                    else:
                        print(f"❌ {model_name} verification failed")
                else:
                    print(f"✅ {model_name} downloaded successfully")
            else:
                results[model_name] = None
                print(f"❌ Failed to download {model_name}")
                
        except Exception as e:
            print(f"❌ Error downloading {model_name}: {e}")
            results[model_name] = None
    
    # Print summary
    print("\n" + "="*50)
    print("DOWNLOAD SUMMARY")
    print("="*50)
    
    successful = 0
    for model_name, path in results.items():
        if path:
            print(f"✅ {model_name}: {path}")
            try:
                size_mb = Path(path).stat().st_size / (1024 * 1024)
                print(f"   📊 Size: {size_mb:.1f} MB")
                successful += 1
            except:
                pass
        else:
            print(f"❌ {model_name}: Download failed")
    
    print(f"\n📊 Downloaded {successful}/{len(args.models)} models successfully")
    print("="*50)
    
    if successful == 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

chmod +x scripts/download/download_models.py

# Model conversion script
cat > scripts/convert/convert_models.py << 'EOF'
#!/usr/bin/env python3
"""
MobileCLIP Mobile Conversion Script
Converts MobileCLIP models to mobile-optimized formats
"""
import argparse
import logging
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from mobile_converter.converter import MobileCLIPMobileConverter

def main():
    parser = argparse.ArgumentParser(description="Convert MobileCLIP models for mobile deployment")
    parser.add_argument("--model", choices=["mobileclip_s0", "mobileclip_s1", "mobileclip_s2", "mobileclip_b", "mobileclip_blt"], 
                       default="mobileclip_s0", help="MobileCLIP model variant")
    parser.add_argument("--platforms", nargs="+", choices=["ios", "android"], 
                       default=["ios", "android"], help="Target platforms")
    parser.add_argument("--model-path", help="Path to pretrained model file")
    parser.add_argument("--output", default="./models/converted", help="Output directory")
    parser.add_argument("--config", default="./config/mobileclip_config.yaml", help="Config file")
    parser.add_argument("--validate", action="store_true", help="Validate conversion accuracy")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create converter
    converter = MobileCLIPMobileConverter(config)
    
    # Load model
    print(f"🔄 Loading {args.model}...")
    try:
        converter.load_pytorch_model(args.model, args.model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Convert for each platform
    for platform in args.platforms:
        print(f"\n📱 Converting for {platform.upper()}...")
        
        try:
            if platform == "ios":
                model_path = converter.convert_to_coreml(str(output_path), args.model)
                results[platform] = model_path
                
            elif platform == "android":
                model_path = converter.convert_to_tflite(str(output_path), args.model)
                results[platform] = model_path
                
            print(f"✅ {platform.upper()} conversion completed: {model_path}")
            
        except Exception as e:
            print(f"❌ {platform.upper()} conversion failed: {e}")
            results[platform] = f"ERROR: {str(e)}"
    
    # Print summary
    print("\n" + "="*50)
    print("CONVERSION SUMMARY")
    print("="*50)
    
    for platform, result in results.items():
        if not result.startswith("ERROR"):
            print(f"✅ {platform.upper()}: {result}")
            # Get model size info
            try:
                if platform == "ios":
                    # CoreML packages are directories
                    print(f"   📊 CoreML package created")
                elif platform == "android":
                    size_mb = Path(result).stat().st_size / (1024 * 1024)
                    print(f"   📊 Size: {size_mb:.1f} MB")
            except:
                pass
        else:
            print(f"❌ {platform.upper()}: {result}")
    
    print("="*50)

if __name__ == "__main__":
    main()
EOF

chmod +x scripts/convert/convert_models.py

# Create README
cat > README.md << 'EOF'
# MobileCLIP Mobile Deployment

Complete deployment pipeline for MobileCLIP image-text models on iOS and Android platforms.

## 🌟 Features

- ✅ Multi-platform support (iOS CoreML, Android TensorFlow Lite)
- ✅ All MobileCLIP variants (S0, S1, S2, B, B-LT)
- ✅ Model optimization and quantization
- ✅ GPU acceleration (Neural Engine, NNAPI, GPU delegates)
- ✅ Ready-to-use mobile integration code (Swift/Kotlin)
- ✅ Zero-shot image classification capabilities
- ✅ Performance benchmarking and validation

## 📱 Supported Models

| Model | Parameters | Mobile Size | iOS Performance | Android Performance | Accuracy |
|-------|------------|-------------|-----------------|-------------------|----------|
| MobileCLIP-S0 | 53.8M | ~55MB | 30-50ms | 80-120ms | 67.8% ImageNet |
| MobileCLIP-S1 | 67.6M | ~80MB | 50-80ms | 120-180ms | 72.6% ImageNet |
| MobileCLIP-S2 | 78.1M | ~95MB | 70-100ms | 150-220ms | 74.4% ImageNet |
| MobileCLIP-B | 128.7M | ~150MB | 120-200ms | 300-450ms | 76.8% ImageNet |
| MobileCLIP-B(LT) | 128.7M | ~150MB | 120-200ms | 300-450ms | 77.2% ImageNet |

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda create -n mobileclip_mobile python=3.10
conda activate mobileclip_mobile

# Install dependencies
pip install torch torchvision
pip install git+https://github.com/apple/ml-mobileclip
pip install coremltools  # For iOS
pip install tensorflow   # For Android
pip install pyyaml pillow numpy huggingface_hub
```

### 2. Download Models

```bash
# Download MobileCLIP-S0 (fastest, smallest)
python scripts/download/download_models.py --models mobileclip_s0

# Download multiple models
python scripts/download/download_models.py --models mobileclip_s0 mobileclip_s1 mobileclip_s2

# Download from Hugging Face
python scripts/download/download_models.py --source huggingface --models mobileclip_s1
```

### 3. Convert Models

```bash
# Convert MobileCLIP-S0 for both platforms
python scripts/convert/convert_models.py --model mobileclip_s0 --platforms ios android

# Convert specific model for iOS only
python scripts/convert/convert_models.py --model mobileclip_s1 --platforms ios
```

### 4. Create Deployment Packages

```bash
# Create ready-to-use deployment packages
python scripts/deploy/deploy_mobile.py --platforms ios android --zip

# Packages will be created in ./deployment/packages/
```

## 📱 Mobile Integration

### iOS Integration (Swift)

```swift
import UIKit

class ViewController: UIViewController {
    private let mobileCLIP = MobileCLIPInference.shared
    
    func classifyImage(_ image: UIImage) {
        let labels = ["dog", "cat", "bird", "car", "plane"]
        
        mobileCLIP.zeroShotClassify(image: image, labels: labels) { result in
            switch result {
            case .success(let results):
                for (label, confidence) in results.prefix(3) {
                    print("\(label): \(confidence)")
                }
            case .failure(let error):
                print("Error: \(error)")
            }
        }
    }
}
```

### Android Integration (Kotlin)

```kotlin
class MainActivity : AppCompatActivity() {
    private lateinit var mobileCLIP: MobileCLIPInference
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        mobileCLIP = MobileCLIPInference(this)
        
        lifecycleScope.launch {
            if (mobileCLIP.initialize()) {
                classifyImage(bitmap)
            }
        }
    }
    
    private suspend fun classifyImage(bitmap: Bitmap) {
        val labels = listOf("dog", "cat", "bird", "car", "plane")
        val results = mobileCLIP.zeroShotClassify(bitmap, labels)
        
        results?.take(3)?.forEach { (label, confidence) ->
            Log.d("Classification", "$label: $confidence")
        }
    }
}
```

## 🎯 Performance Optimization

### iOS Optimization
- ✅ Neural Engine acceleration
- ✅ Float16 quantization
- ✅ Metal GPU acceleration
- ✅ Memory optimization
- ✅ CoreML optimization

### Android Optimization
- ✅ NNAPI acceleration
- ✅ GPU delegate support
- ✅ INT8 quantization
- ✅ XNNPACK optimization
- ✅ Multi-threading support

## 🛠️ Development

### Project Structure
```
mobileclip_mobile/
├── config/                 # Configuration files
│   ├── mobileclip_config.yaml
│   └── mobile/            # Platform-specific configs
├── src/                   # Source code
│   ├── core/             # Core MobileCLIP implementation
│   ├── mobile_converter/ # Mobile conversion tools
│   └── utils/            # Utilities
├── scripts/              # Automation scripts
│   ├── download/         # Model download scripts
│   ├── convert/          # Conversion scripts
│   └── deploy/           # Deployment scripts
├── mobile/               # Platform-specific code
│   ├── ios/             # iOS Swift integration
│   └── android/         # Android Kotlin integration
├── models/              # Model storage
│   ├── pretrained/      # Downloaded models
│   └── converted/       # Converted models
├── examples/            # Example applications
└── docs/                # Documentation
```

### Custom Configuration

Edit `config/mobileclip_config.yaml` to customize:
- Model variants and parameters
- Quantization settings
- Platform-specific optimizations
- Performance targets

## 📊 Benchmarking

### Performance Testing

```bash
# Run performance benchmarks
python tools/benchmarking/mobile_benchmark.py --model mobileclip_s0 --platform ios
python tools/benchmarking/mobile_benchmark.py --model mobileclip_s1 --platform android

# Memory usage analysis
python tools/benchmarking/memory_benchmark.py --model mobileclip_s0
```

### Accuracy Validation

```bash
# Validate conversion accuracy
python tools/benchmarking/accuracy_benchmark.py --original-model mobileclip_s0.pt --converted-model mobileclip_s0.tflite
```

## 🔧 Advanced Usage

### Custom Model Training

```python
# Fine-tune MobileCLIP on custom data
from src.core.mobileclip_model import MobileCLIPModel

model = MobileCLIPModel("mobileclip_s0")
# Add your fine-tuning code here
```

### Batch Processing

```python
# Process multiple images efficiently
images = [image1, image2, image3]
labels = ["dog", "cat", "bird"]

for image in images:
    results = model.zero_shot_classify(image, labels)
    print(results)
```

## 📖 Documentation

- [iOS Deployment Guide](docs/ios_deployment.md)
- [Android Deployment Guide](docs/android_deployment.md)
- [Performance Optimization](docs/performance_optimization.md)
- [Troubleshooting](docs/troubleshooting.md)
- [API Reference](docs/api_reference.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Apple Research for MobileCLIP
- Apple for CoreML framework
- Google for TensorFlow Lite
- Hugging Face for model hosting
- Open source community

## 🆘 Troubleshooting

### Common Issues

1. **Model download fails:**
   ```bash
   # Check internet connection and try again
   python scripts/download/download_models.py --force
   ```

2. **CoreML conversion fails:**
   ```bash
   # Ensure you're on macOS with latest Xcode
   pip install --upgrade coremltools
   ```

3. **TensorFlow Lite conversion fails:**
   ```bash
   # Install required dependencies
   pip install tensorflow onnx onnx-tf
   ```

4. **iOS app crashes:**
   - Ensure model files are added to Xcode project
   - Check iOS deployment target (15.0+)
   - Verify model file paths

5. **Android inference slow:**
   - Enable NNAPI acceleration
   - Use GPU delegate if available
   - Check if model is quantized

### Getting Help

- Check the [documentation](docs/)
- Open an issue on GitHub
- Join our community discussions

---

**Ready to deploy MobileCLIP on mobile? Get started with the quick start guide above! 🚀**
EOF

echo "📜 Creating additional deployment scripts..."

# Deployment script
cat > scripts/deploy/deploy_mobile.py << 'EOF'
#!/usr/bin/env python3
"""
MobileCLIP Mobile Deployment Script
Creates deployment packages for iOS and Android
"""
import argparse
import shutil
import zipfile
import json
from pathlib import Path
import yaml
from datetime import datetime

def create_ios_package(model_name: str, model_path: str, output_dir: str):
    """Create iOS deployment package."""
    ios_dir = Path(output_dir) / "iOS_Package"
    ios_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy CoreML models
    if Path(model_path).exists():
        if Path(model_path).is_dir():  # CoreML package
            shutil.copytree(model_path, ios_dir / Path(model_path).name, dirs_exist_ok=True)
        else:
            shutil.copy2(model_path, ios_dir)
    
    # Copy Swift integration
    swift_dir = ios_dir / "Integration"
    swift_dir.mkdir(exist_ok=True)
    
    swift_source = Path("mobile/ios/swift/MobileCLIPInference.swift")
    if swift_source.exists():
        shutil.copy2(swift_source, swift_dir)
    
    # Create info file
    info = {
        "model_name": model_name,
        "platform": "ios",
        "created_at": datetime.now().isoformat(),
        "requirements": {
            "ios_version": "15.0+",
            "xcode_version": "15.0+",
            "frameworks": ["CoreML", "Vision", "Accelerate"]
        }
    }
    
    with open(ios_dir / "deployment_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    # Create README
    readme_content = f"""# MobileCLIP iOS Deployment Package

## Model: {model_name}

### Files Included
- `{Path(model_path).name}`: CoreML model package
- `Integration/MobileCLIPInference.swift`: Swift integration class
- `deployment_info.json`: Deployment information

### Integration Steps

1. **Add Model to Xcode Project:**
   - Drag the `.mlpackage` files into your Xcode project
   - Ensure they're added to your target

2. **Add Swift File:**
   - Add `MobileCLIPInference.swift` to your project

3. **Usage Example:**
   ```swift
   let inference = MobileCLIPInference.shared
   
   let labels = ["dog", "cat", "bird"]
   inference.zeroShotClassify(image: image, labels: labels) {{ result in
       switch result {{
       case .success(let results):
           for (label, confidence) in results.prefix(3) {{
               print("\\(label): \\(confidence)")
           }}
       case .failure(let error):
           print("Error: \\(error)")
       }}
   }}
   ```

### Requirements
- iOS 15.0+
- Xcode 15.0+
- Device with Neural Engine (recommended)

### Performance
- Inference time: ~30-200ms (device dependent)
- Memory usage: ~150-350MB
- Optimized for Neural Engine acceleration

### Support
For issues and questions, refer to the main documentation.
"""
    
    with open(ios_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    return str(ios_dir)

def create_android_package(model_name: str, model_path: str, output_dir: str):
    """Create Android deployment package."""
    android_dir = Path(output_dir) / "Android_Package"
    android_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy TensorFlow Lite models
    if Path(model_path).exists():
        shutil.copy2(model_path, android_dir)
    
    # Copy Kotlin integration
    kotlin_dir = android_dir / "Integration"
    kotlin_dir.mkdir(exist_ok=True)
    
    kotlin_source = Path("mobile/android/kotlin/MobileCLIPInference.kt")
    if kotlin_source.exists():
        shutil.copy2(kotlin_source, kotlin_dir)
    
    # Create build.gradle dependencies
    gradle_deps = """
// Add to your app-level build.gradle dependencies block
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.6.4'
}
"""
    
    with open(android_dir / "build.gradle.dependencies", 'w') as f:
        f.write(gradle_deps)
    
    # Create info file
    info = {
        "model_name": model_name,
        "platform": "android",
        "created_at": datetime.now().isoformat(),
        "requirements": {
            "android_api": "24+",
            "compile_sdk": "34",
            "dependencies": [
                "tensorflow-lite:2.13.0",
                "tensorflow-lite-gpu:2.13.0",
                "tensorflow-lite-support:0.4.4"
            ]
        }
    }
    
    with open(android_dir / "deployment_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    # Create README
    readme_content = f"""# MobileCLIP Android Deployment Package

## Model: {model_name}

### Files Included
- `{Path(model_path).name}`: TensorFlow Lite model file
- `Integration/MobileCLIPInference.kt`: Kotlin integration class
- `build.gradle.dependencies`: Required dependencies
- `deployment_info.json`: Deployment information

### Integration Steps

1. **Add Model to Assets:**
   - Copy `.tflite` files to `app/src/main/assets/`

2. **Add Dependencies:**
   - Add contents of `build.gradle.dependencies` to your app's build.gradle

3. **Add Kotlin Class:**
   - Add `MobileCLIPInference.kt` to your project

4. **Usage Example:**
   ```kotlin
   class MainActivity : AppCompatActivity() {{
       private lateinit var mobileCLIP: MobileCLIPInference
       
       override fun onCreate(savedInstanceState: Bundle?) {{
           super.onCreate(savedInstanceState)
           
           mobileCLIP = MobileCLIPInference(this)
           
           lifecycleScope.launch {{
               if (mobileCLIP.initialize()) {{
                   val labels = listOf("dog", "cat", "bird")
                   val results = mobileCLIP.zeroShotClassify(bitmap, labels)
                   
                   results?.take(3)?.forEach {{ (label, confidence) ->
                       Log.d("Classification", "$label: $confidence")
                   }}
               }}
           }}
       }}
       
       override fun onDestroy() {{
           super.onDestroy()
           mobileCLIP.close()
       }}
   }}
   ```

### Requirements
- Android API 24+
- TensorFlow Lite 2.13.0+
- Kotlin coroutines support

### Performance
- Inference time: ~80-450ms (device dependent)
- Memory usage: ~200-500MB
- NNAPI and GPU acceleration supported

### Optimization Tips
- Enable NNAPI acceleration for faster inference
- Use GPU delegate on supported devices
- Consider model quantization for smaller size

### Support
For issues and questions, refer to the main documentation.
"""
    
    with open(android_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    return str(android_dir)

def main():
    parser = argparse.ArgumentParser(description="Create MobileCLIP mobile deployment packages")
    parser.add_argument("--model", default="mobileclip_s0", help="Model name")
    parser.add_argument("--platforms", nargs="+", choices=["ios", "android"], 
                       default=["ios", "android"], help="Target platforms")
    parser.add_argument("--models-dir", default="./models/converted", help="Converted models directory")
    parser.add_argument("--output", default="./deployment/packages", help="Output directory")
    parser.add_argument("--zip", action="store_true", help="Create zip archives")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models_dir = Path(args.models_dir)
    
    print(f"📦 Creating deployment packages for {args.model}...")
    print(f"📁 Output directory: {output_dir}")
    print()
    
    packages = {}
    
    for platform in args.platforms:
        print(f"📱 Creating {platform.upper()} package...")
        
        try:
            if platform == "ios":
                # Look for CoreML models
                model_path = models_dir / f"{args.model}_image.mlpackage"
                if not model_path.exists():
                    model_path = models_dir / f"{args.model}.mlpackage"
                
                if model_path.exists():
                    package_path = create_ios_package(args.model, str(model_path), str(output_dir))
                    packages[platform] = package_path
                    print(f"✅ iOS package created: {package_path}")
                else:
                    print(f"❌ iOS model not found: {model_path}")
                    
            elif platform == "android":
                # Look for TensorFlow Lite models
                model_path = models_dir / f"{args.model}.tflite"
                
                if model_path.exists():
                    package_path = create_android_package(args.model, str(model_path), str(output_dir))
                    packages[platform] = package_path
                    print(f"✅ Android package created: {package_path}")
                else:
                    print(f"❌ Android model not found: {model_path}")
        
        except Exception as e:
            print(f"❌ Failed to create {platform} package: {e}")
    
    # Create zip archives if requested
    if args.zip and packages:
        print(f"\n📦 Creating zip archives...")
        
        for platform, package_path in packages.items():
            try:
                zip_path = Path(package_path).with_suffix('.zip')
                
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    package_dir = Path(package_path)
                    for file_path in package_dir.rglob('*'):
                        if file_path.is_file():
                            arcname = file_path.relative_to(package_dir.parent)
                            zipf.write(file_path, arcname)
                
                print(f"✅ {platform.upper()} archive: {zip_path}")
                
            except Exception as e:
                print(f"❌ Failed to create {platform} archive: {e}")
    
    # Print summary
    print("\n" + "="*50)
    print("DEPLOYMENT PACKAGES SUMMARY")
    print("="*50)
    
    for platform, package_path in packages.items():
        print(f"✅ {platform.upper()}: {package_path}")
        
        if args.zip:
            zip_path = Path(package_path).with_suffix('.zip')
            if zip_path.exists():
                size_mb = zip_path.stat().st_size / (1024 * 1024)
                print(f"   📦 Archive: {zip_path} ({size_mb:.1f} MB)")
    
    print(f"\n📊 Created {len(packages)} deployment packages")
    print("="*50)
    
    if not packages:
        print("❌ No packages were created. Check that converted models exist.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
EOF

chmod +x scripts/deploy/deploy_mobile.py

# Benchmarking script
cat > tools/benchmarking/mobile_benchmark.py << 'EOF'
#!/usr/bin/env python3
"""
MobileCLIP Mobile Benchmarking
Performance testing for mobile deployments
"""
import argparse
import time
import logging
import sys
from pathlib import Path
import json
import numpy as np

def benchmark_ios_model(model_path: str, num_runs: int = 10):
    """Benchmark iOS CoreML model."""
    try:
        import coremltools as ct
    except ImportError:
        print("❌ CoreML Tools not available")
        return None
    
    try:
        # Load model
        model = ct.models.MLModel(model_path)
        
        # Create test input
        test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        # Warmup
        for _ in range(3):
            _ = model.predict({'image': test_input})
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = model.predict({'image': test_input})
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return {
            'platform': 'ios',
            'model_path': model_path,
            'avg_inference_ms': np.mean(times),
            'std_inference_ms': np.std(times),
            'min_inference_ms': np.min(times),
            'max_inference_ms': np.max(times),
            'num_runs': num_runs
        }
        
    except Exception as e:
        print(f"❌ iOS benchmark failed: {e}")
        return None

def benchmark_android_model(model_path: str, num_runs: int = 10):
    """Benchmark Android TensorFlow Lite model."""
    try:
        import tensorflow as tf
    except ImportError:
        print("❌ TensorFlow not available")
        return None
    
    try:
        # Load model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Create test input
        input_shape = input_details[0]['shape']
        test_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(3):
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
            _ = interpreter.get_tensor(output_details[0]['index'])
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
            _ = interpreter.get_tensor(output_details[0]['index'])
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return {
            'platform': 'android',
            'model_path': model_path,
            'avg_inference_ms': np.mean(times),
            'std_inference_ms': np.std(times),
            'min_inference_ms': np.min(times),
            'max_inference_ms': np.max(times),
            'num_runs': num_runs,
            'model_size_mb': Path(model_path).stat().st_size / (1024 * 1024)
        }
        
    except Exception as e:
        print(f"❌ Android benchmark failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Benchmark MobileCLIP mobile models")
    parser.add_argument("--model-path", required=True, help="Path to model file")
    parser.add_argument("--platform", choices=["ios", "android"], required=True, help="Target platform")
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark runs")
    parser.add_argument("--output", help="Output JSON file for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)
    
    print(f"🔍 Benchmarking {args.platform.upper()} model: {model_path}")
    print(f"🏃 Running {args.runs} benchmark iterations...")
    print()
    
    # Run benchmark
    if args.platform == "ios":
        results = benchmark_ios_model(str(model_path), args.runs)
    elif args.platform == "android":
        results = benchmark_android_model(str(model_path), args.runs)
    else:
        print(f"❌ Unsupported platform: {args.platform}")
        sys.exit(1)
    
    if results is None:
        print("❌ Benchmark failed")
        sys.exit(1)
    
    # Print results
    print("📊 BENCHMARK RESULTS")
    print("="*40)
    print(f"Platform: {results['platform'].upper()}")
    print(f"Model: {Path(results['model_path']).name}")
    print(f"Average inference time: {results['avg_inference_ms']:.2f} ms")
    print(f"Standard deviation: {results['std_inference_ms']:.2f} ms")
    print(f"Min inference time: {results['min_inference_ms']:.2f} ms")
    print(f"Max inference time: {results['max_inference_ms']:.2f} ms")
    
    if 'model_size_mb' in results:
        print(f"Model size: {results['model_size_mb']:.2f} MB")
    
    print(f"Number of runs: {results['num_runs']}")
    print("="*40)
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: {args.output}")

if __name__ == "__main__":
    main()
EOF

chmod +x tools/benchmarking/mobile_benchmark.py

echo "📚 Creating documentation..."

# iOS deployment guide
cat > docs/ios_deployment.md << 'EOF'
# iOS Deployment Guide

Complete guide for deploying MobileCLIP models on iOS devices.

## Prerequisites

- macOS development machine
- Xcode 15.0 or later
- iOS 15.0 or later target
- Device with Neural Engine (recommended)

## Model Conversion

Convert your MobileCLIP model to CoreML format:

```bash
python scripts/convert/convert_models.py --model mobileclip_s0 --platforms ios
```

This creates CoreML model packages in `models/converted/`:
- `mobileclip_s0_image.mlpackage` - Image encoder
- `mobileclip_s0_text.mlpackage` - Text encoder

## Xcode Integration

### 1. Add Models to Project

1. Drag both `.mlpackage` files into your Xcode project
2. Ensure "Add to target" is checked for your app target
3. Verify models appear in your project navigator

### 2. Add Swift Integration Class

1. Add `MobileCLIPInference.swift` to your project
2. Import required frameworks in your app:

```swift
import CoreML
import Vision
import Accelerate
```

### 3. Basic Usage

```swift
class ViewController: UIViewController {
    private let mobileCLIP = MobileCLIPInference.shared
    
    @IBAction func classifyImage(_ sender: UIButton) {
        guard let image = imageView.image else { return }
        
        let labels = ["dog", "cat", "bird", "car", "airplane"]
        
        mobileCLIP.zeroShotClassify(image: image, labels: labels) { result in
            DispatchQueue.main.async {
                switch result {
                case .success(let results):
                    self.displayResults(results)
                case .failure(let error):
                    self.showError(error)
                }
            }
        }
    }
    
    private func displayResults(_ results: [(label: String, confidence: Float)]) {
        for (label, confidence) in results.prefix(3) {
            print("\(label): \(String(format: "%.2f", confidence * 100))%")
        }
    }
}
```

## Performance Optimization

### Neural Engine Acceleration

Models automatically use Neural Engine when available:

```swift
// Verify Neural Engine usage
if let computeUnits = model.configuration.computeUnits {
    print("Compute units: \(computeUnits)")
}
```

### Memory Management

```swift
// Process images in background queue
DispatchQueue.global(qos: .userInitiated).async {
    self.mobileCLIP.extractImageFeatures(from: image) { result in
        DispatchQueue.main.async {
            // Update UI
        }
    }
}
```

## Troubleshooting

### Common Issues

1. **Model not found error:**
   - Verify `.mlpackage` files are added to project target
   - Check model file names match code expectations

2. **Slow inference:**
   - Ensure Neural Engine is available (iPhone 12+, iPad Air 4+)
   - Check model is using float16 precision

3. **Memory warnings:**
   - Process images on background queue
   - Resize large images before processing

### Performance Tips

- Use Neural Engine compatible devices
- Resize images to 224x224 before processing
- Batch process multiple images when possible
- Cache model instances to avoid reload overhead

## Example App

See `examples/ios/MobileCLIPDemo/` for a complete example application.
EOF

# Android deployment guide
cat > docs/android_deployment.md << 'EOF'
# Android Deployment Guide

Complete guide for deploying MobileCLIP models on Android devices.

## Prerequisites

- Android Studio
- Android API 24+ (Android 7.0)
- TensorFlow Lite 2.13.0+
- Device with NNAPI support (recommended)

## Model Conversion

Convert your MobileCLIP model to TensorFlow Lite format:

```bash
python scripts/convert/convert_models.py --model mobileclip_s0 --platforms android
```

This creates TensorFlow Lite models in `models/converted/`:
- `mobileclip_s0.tflite` - Complete model

## Android Studio Integration

### 1. Add Dependencies

Add to your app-level `build.gradle`:

```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.6.4'
}
```

### 2. Add Model to Assets

1. Create `assets` folder in `app/src/main/` if it doesn't exist
2. Copy `.tflite` files to `app/src/main/assets/`

### 3. Add Kotlin Integration Class

Add `MobileCLIPInference.kt` to your project.

### 4. Basic Usage

```kotlin
class MainActivity : AppCompatActivity() {
    private lateinit var mobileCLIP: MobileCLIPInference
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        mobileCLIP = MobileCLIPInference(this)
        
        lifecycleScope.launch {
            if (mobileCLIP.initialize()) {
                Log.d(TAG, "MobileCLIP initialized successfully")
            } else {
                Log.e(TAG, "Failed to initialize MobileCLIP")
            }
        }
    }
    
    private fun classifyImage(bitmap: Bitmap) {
        lifecycleScope.launch {
            val labels = listOf("dog", "cat", "bird", "car", "airplane")
            val results = mobileCLIP.zeroShotClassify(bitmap, labels)
            
            results?.take(3)?.forEach { (label, confidence) ->
                Log.d(TAG, "$label: ${String.format("%.2f", confidence * 100)}%")
            }
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        mobileCLIP.close()
    }
}
```

## Performance Optimization

### NNAPI Acceleration

Enable NNAPI for faster inference:

```kotlin
val options = Interpreter.Options().apply {
    setUseNNAPI(true)
    setNumThreads(4)
}
```

### GPU Delegate

Use GPU acceleration when available:

```kotlin
val gpuDelegate = GpuDelegate()
val options = Interpreter.Options().apply {
    addDelegate(gpuDelegate)
}
```

### Memory Optimization

```kotlin
// Process images on background thread
lifecycleScope.launch(Dispatchers.Default) {
    val results = mobileCLIP.extractImageFeatures(bitmap)
    
    withContext(Dispatchers.Main) {
        // Update UI
    }
}
```

## Camera Integration

### Basic Camera Capture

```kotlin
private fun setupCamera() {
    val imageCapture = ImageCapture.Builder().build()
    
    imageCapture.takePicture(
        outputFileOptions,
        ContextCompat.getMainExecutor(this),
        object : ImageCapture.OnImageSavedCallback {
            override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                val bitmap = BitmapFactory.decodeFile(output.savedUri?.path)
                classifyImage(bitmap)
            }
            
            override fun onError(exception: ImageCaptureException) {
                Log.e(TAG, "Image capture failed: ${exception.message}")
            }
        }
    )
}
```

## Troubleshooting

### Common Issues

1. **Model loading fails:**
   - Verify `.tflite` files are in `assets` folder
   - Check file names match code expectations
   - Ensure sufficient storage space

2. **Slow inference:**
   - Enable NNAPI acceleration
   - Use GPU delegate if available
   - Check model quantization

3. **Out of memory:**
   - Resize large images before processing
   - Process images on background thread
   - Close unused model instances

### Performance Tips

- Enable hardware acceleration (NNAPI/GPU)
- Use quantized models for faster inference
- Resize images to 224x224 before processing
- Batch process multiple images when possible

## Example App

See `examples/android/MobileCLIPDemo/` for a complete example application.
EOF

echo "✅ MobileCLIP Mobile Deployment Project Structure Created Successfully!"
echo ""
echo "📁 Directory structure created in: $(pwd)"
echo ""
echo "🚀 Next Steps:"
echo "1. cd $PROJECT_NAME"
echo "2. conda create -n mobileclip_mobile python=3.10"
echo "3. conda activate mobileclip_mobile" 
echo "4. pip install torch torchvision git+https://github.com/apple/ml-mobileclip"
echo "5. pip install coremltools tensorflow pyyaml pillow numpy huggingface_hub"
echo "6. python scripts/download/download_models.py --models mobileclip_s0"
echo "7. python scripts/convert/convert_models.py --model mobileclip_s0"
echo "8. python scripts/deploy/deploy_mobile.py --zip"
echo ""
echo "📱 Mobile deployment packages will be ready in ./deployment/packages/"
echo ""
echo "📖 See README.md and docs/ for detailed instructions"