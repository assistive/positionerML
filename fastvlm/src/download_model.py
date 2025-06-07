#!/usr/bin/env python3
# fastvlm/scripts/download_model.py

import argparse
import os
import sys
from pathlib import Path
import requests
import torch
from tqdm import tqdm
import hashlib
import json
import logging
from typing import Optional, Dict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.fastvlm_model import FastVLMModel, FastVLMConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model registry
MODEL_REGISTRY = {
    "fastvlm-tiny": {
        "url": "https://huggingface.co/apple/fastvlm-tiny/resolve/main/",
        "files": ["pytorch_model.bin", "config.json", "tokenizer.json"],
        "size_mb": 200,
        "description": "Tiny model optimized for mobile devices"
    },
    "fastvlm-small": {
        "url": "https://huggingface.co/apple/fastvlm-small/resolve/main/",
        "files": ["pytorch_model.bin", "config.json", "tokenizer.json"],
        "size_mb": 500,
        "description": "Small model with good balance of speed and accuracy"
    },
    "fastvlm-base": {
        "url": "https://huggingface.co/apple/fastvlm-base/resolve/main/",
        "files": ["pytorch_model.bin", "config.json", "tokenizer.json"],
        "size_mb": 1200,
        "description": "Base model for general use cases"
    },
    "fastvlm-large": {
        "url": "https://huggingface.co/apple/fastvlm-large/resolve/main/",
        "files": ["pytorch_model.bin", "config.json", "tokenizer.json"],
        "size_mb": 3000,
        "description": "Large model for best performance"
    }
}


def download_file(url: str, output_path: Path, expected_size: Optional[int] = None) -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as file:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
                    pbar.update(len(chunk))
        
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def verify_file(file_path: Path, expected_hash: Optional[str] = None) -> bool:
    """Verify file integrity."""
    if not file_path.exists():
        return False
    
    if expected_hash:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        actual_hash = sha256_hash.hexdigest()
        return actual_hash == expected_hash
    
    return True


def download_model(model_name: str, output_dir: str, force: bool = False) -> Path:
    """Download a FastVLM model."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")
    
    model_info = MODEL_REGISTRY[model_name]
    output_path = Path(output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading {model_name} to {output_path}")
    logger.info(f"Description: {model_info['description']}")
    logger.info(f"Estimated size: {model_info['size_mb']} MB")
    
    # Check if already downloaded
    if not force and all((output_path / file).exists() for file in model_info['files']):
        logger.info(f"Model {model_name} already exists. Use --force to re-download.")
        return output_path
    
    # Download files
    success = True
    for file_name in model_info['files']:
        file_url = model_info['url'] + file_name
        file_path = output_path / file_name
        
        if not download_file(file_url, file_path):
            success = False
            break
    
    if success:
        logger.info(f"Successfully downloaded {model_name}")
        
        # Create model info file
        info_data = {
            "model_name": model_name,
            "download_date": str(Path.ctime(output_path)),
            "files": model_info['files'],
            "size_mb": model_info['size_mb']
        }
        
        with open(output_path / "model_info.json", 'w') as f:
            json.dump(info_data, f, indent=2)
    else:
        logger.error(f"Failed to download {model_name}")
        # Clean up partial downloads
        import shutil
        shutil.rmtree(output_path)
        raise RuntimeError(f"Failed to download {model_name}")
    
    return output_path


def create_model_from_download(model_path: Path) -> FastVLMModel:
    """Create a FastVLM model from downloaded files."""
    logger.info(f"Loading model from {model_path}")
    
    # Load config
    config_path = model_path / "config.json"
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = FastVLMConfig(**config_dict)
    
    # Load model
    model = FastVLMModel(config)
    
    # Load weights
    weights_path = model_path / "pytorch_model.bin"
    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict)
        logger.info("Model weights loaded successfully")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Download FastVLM models")
    parser.add_argument(
        "--model",
        type=str,
        default="fastvlm-base",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to download"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/pretrained",
        help="Output directory for downloaded models"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model exists"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify downloaded model"
    )
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Available FastVLM models:")
        print("-" * 60)
        for name, info in MODEL_REGISTRY.items():
            print(f"{name:15} | {info['size_mb']:>6} MB | {info['description']}")
        return
    
    try:
        # Download model
        model_path = download_model(args.model, args.output, args.force)
        
        # Verify model if requested
        if args.verify:
            logger.info("Verifying downloaded model...")
            try:
                model = create_model_from_download(model_path)
                logger.info("Model verification successful!")
                
                # Print model info
                print(f"\nModel Information:")
                print(f"  Architecture: {model.config.model_type}")
                print(f"  Vision encoder: {model.config.vision_encoder}")
                print(f"  Language model: {model.config.language_model}")
                print(f"  Hidden size: {model.config.hidden_size}")
                print(f"  Vision tokens: {model.config.num_vision_tokens}")
                
                # Count parameters
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"  Total parameters: {total_params:,}")
                print(f"  Trainable parameters: {trainable_params:,}")
                
            except Exception as e:
                logger.error(f"Model verification failed: {e}")
                sys.exit(1)
        
        print(f"\nModel downloaded successfully to: {model_path}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
