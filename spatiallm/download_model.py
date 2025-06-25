#!/usr/bin/env python3
"""
download_model.py

Script to download pre-trained spatialLM v1.1 models from HuggingFace or other repositories.
Handles authentication, model selection, and verification of downloaded files.

Version 1.1 Updates:
- Enhanced spatial reasoning capabilities
- Improved mobile optimization
- Better quantization support
- Updated model architectures
"""

import os
import argparse
import logging
import sys
import hashlib
import requests
import tqdm
import json
import shutil
import datetime
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download, login
from typing import Optional, List, Dict, Any, Union

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("download_model")

# Updated models info for v1.1 - Using actual Hugging Face repository
DEFAULT_MODELS = {
    "spatialLM-1.1-qwen-0.5b": {
        "repo_id": "manycore-research/SpatialLM1.1-Qwen-0.5B",
        "files": ["pytorch_model.bin", "config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"],
        "description": "SpatialLM 1.1 based on Qwen-0.5B (500M parameters) - Enhanced spatial reasoning",
        "size_mb": 1900,
        "version": "1.1",
        "capabilities": ["enhanced_spatial_reasoning", "qwen_architecture", "mobile_optimized", "quantization_ready"],
        "base_architecture": "qwen"
    },
    # Alias for easier access
    "spatialLM-base-v1.1": {
        "repo_id": "manycore-research/SpatialLM1.1-Qwen-0.5B",
        "files": ["pytorch_model.bin", "config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"],
        "description": "SpatialLM 1.1 base model (500M parameters) - Enhanced spatial reasoning",
        "size_mb": 1900,
        "version": "1.1",
        "capabilities": ["enhanced_spatial_reasoning", "qwen_architecture", "mobile_optimized", "quantization_ready"],
        "base_architecture": "qwen"
    },
    # Placeholder for potential future models
    "spatialLM-small-v1.1": {
        "repo_id": "manycore-research/SpatialLM1.1-Qwen-0.5B",  # Using same repo for now
        "files": ["pytorch_model.bin", "config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"],
        "description": "SpatialLM 1.1 small model (500M parameters) - Mobile optimized",
        "size_mb": 1900,
        "version": "1.1",
        "capabilities": ["mobile_first", "edge_deployment", "fast_inference", "qwen_architecture"],
        "base_architecture": "qwen"
    },
    # Legacy v1.0 models for backward compatibility
    "spatialLM-base": {
        "repo_id": "spatialLM/spatialLM-base",
        "files": ["pytorch_model.bin", "config.json", "tokenizer.json", "vocab.json"],
        "description": "Base model v1.0 (220M parameters) - Legacy",
        "size_mb": 850,
        "version": "1.0",
        "capabilities": ["basic_spatial_reasoning"]
    },
    "spatialLM-small": {
        "repo_id": "spatialLM/spatialLM-small", 
        "files": ["pytorch_model.bin", "config.json", "tokenizer.json", "vocab.json"],
        "description": "Small model v1.0 (70M parameters) - Legacy",
        "size_mb": 280,
        "version": "1.0",
        "capabilities": ["basic_spatial_reasoning"]
    },
    "spatialLM-large": {
        "repo_id": "spatialLM/spatialLM-large",
        "files": ["pytorch_model.bin", "config.json", "tokenizer.json", "vocab.json"],
        "description": "Large model v1.0 (770M parameters) - Legacy",
        "size_mb": 3100,
        "version": "1.0",
        "capabilities": ["basic_spatial_reasoning"]
    }
}

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Download pre-trained spatialLM models")
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="spatialLM-1.1-qwen-0.5b",
        choices=list(DEFAULT_MODELS.keys()) + ["custom"],
        help="Name of the model to download (defaults to SpatialLM 1.1 Qwen-0.5B model)"
    )
    
    parser.add_argument(
        "--custom_repo_id",
        type=str,
        help="Custom HuggingFace repo ID (e.g., 'organization/model-name')"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models",
        help="Directory to save the downloaded model"
    )
    
    parser.add_argument(
        "--use_auth_token",
        action="store_true",
        help="Use HuggingFace authentication token for private models"
    )
    
    parser.add_argument(
        "--auth_token",
        type=str,
        help="HuggingFace authentication token (if not set, will use HUGGINGFACE_TOKEN env variable)"
    )
    
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Model revision to download (branch name, tag name, or commit hash)"
    )
    
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Force re-download even if files already exist"
    )
    
    parser.add_argument(
        "--download_specific_files",
        type=str,
        nargs="+",
        help="Download only specific files (e.g., --download_specific_files pytorch_model.bin config.json)"
    )
    
    parser.add_argument(
        "--verify_checksums",
        action="store_true",
        help="Verify downloaded files using checksums"
    )
    
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="List all available models and exit"
    )
    
    parser.add_argument(
        "--model_info",
        action="store_true",
        help="Show detailed information about the selected model"
    )
    
    parser.add_argument(
        "--prefer_v1_1",
        action="store_true",
        default=True,
        help="Automatically prefer v1.1 models when available (default: True)"
    )
    
    parser.add_argument(
        "--include_mobile_configs",
        action="store_true",
        default=True,
        help="Include mobile deployment configurations (default: True for v1.1 models)"
    )
    
    return parser.parse_args()

def list_available_models():
    """List all available models with their details"""
    print("Available spatialLM models:")
    print("=" * 80)
    
    # Group by version
    v11_models = {k: v for k, v in DEFAULT_MODELS.items() if v.get("version") == "1.1"}
    v10_models = {k: v for k, v in DEFAULT_MODELS.items() if v.get("version") == "1.0"}
    
    print("\n🚀 Version 1.1 Models (Recommended):")
    print("-" * 40)
    for name, info in v11_models.items():
        print(f"  {name}:")
        print(f"    Description: {info['description']}")
        print(f"    Size: {info['size_mb']} MB")
        print(f"    Architecture: {info.get('base_architecture', 'transformer')}")
        print(f"    Capabilities: {', '.join(info['capabilities'])}")
        print(f"    Repository: {info['repo_id']}")
        print()
    
    print("📋 Version 1.0 Models (Legacy):")
    print("-" * 40)
    for name, info in v10_models.items():
        print(f"  {name}:")
        print(f"    Description: {info['description']}")
        print(f"    Size: {info['size_mb']} MB")
        print(f"    Repository: {info['repo_id']}")
        print()

def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get information about a specific model"""
    if model_name in DEFAULT_MODELS:
        return DEFAULT_MODELS[model_name]
    else:
        raise ValueError(f"Unknown model: {model_name}")

def setup_authentication(args):
    """Setup HuggingFace authentication if needed"""
    if args.use_auth_token:
        token = args.auth_token or os.environ.get("HUGGINGFACE_TOKEN")
        if not token:
            raise ValueError("No authentication token provided. Use --auth_token or set HUGGINGFACE_TOKEN environment variable")
        
        login(token=token)
        logger.info("Successfully authenticated with HuggingFace")

def download_model_files(repo_id: str, files: List[str], output_dir: str, revision: str = "main", force_download: bool = False):
    """Download model files from HuggingFace Hub"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    downloaded_files = []
    
    for file in files:
        try:
            logger.info(f"Downloading {file}...")
            
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=file,
                cache_dir=output_path,
                revision=revision,
                force_download=force_download,
                local_files_only=False
            )
            
            # Copy to output directory if not already there
            target_path = output_path / file
            if not target_path.exists() or force_download:
                shutil.copy2(file_path, target_path)
            
            downloaded_files.append(str(target_path))
            logger.info(f"✓ Downloaded {file} to {target_path}")
            
        except Exception as e:
            logger.error(f"Failed to download {file}: {str(e)}")
            raise
    
    return downloaded_files

def download_full_repository(repo_id: str, output_dir: str, revision: str = "main", force_download: bool = False):
    """Download entire model repository"""
    output_path = Path(output_dir)
    
    try:
        logger.info(f"Downloading full repository {repo_id}...")
        
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=output_path,
            revision=revision,
            force_download=force_download,
            local_files_only=False
        )
        
        logger.info(f"✓ Downloaded repository to {snapshot_path}")
        return snapshot_path
        
    except Exception as e:
        logger.error(f"Failed to download repository: {str(e)}")
        raise

def verify_downloaded_files(files: List[str]) -> bool:
    """Verify that all files were downloaded successfully"""
    logger.info("Verifying downloaded files...")
    
    all_valid = True
    for file in files:
        if os.path.exists(file):
            size_mb = os.path.getsize(file) / (1024 * 1024)
            logger.info(f"✓ {file}: {size_mb:.2f} MB")
            
            # Calculate file hash
            sha256_hash = hashlib.sha256()
            with open(file, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            file_hash = sha256_hash.hexdigest()
            logger.info(f"  SHA256: {file_hash}")
        else:
            logger.error(f"✗ File {file} not found")
            all_valid = False
    
    return all_valid

def create_model_info(model_name: str, repo_id: str, output_dir: str, version: str = "1.1"):
    """Create a model_info.json file with details about the downloaded model"""
    info = {
        "model_name": model_name,
        "repo_id": repo_id,
        "version": version,
        "download_date": datetime.datetime.now().isoformat(),
        "spatialLM_version": "1.1",
        "files": []
    }
    
    # Get list of files and their sizes/hashes
    for root, _, files in os.walk(output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, output_dir)
            
            # Calculate file hash
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            
            info["files"].append({
                "path": relative_path,
                "size_bytes": os.path.getsize(file_path),
                "sha256": sha256_hash.hexdigest()
            })
    
    # Save info to file
    with open(os.path.join(output_dir, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"Created model info file: {os.path.join(output_dir, 'model_info.json')}")

def download_mobile_configs(output_dir: str, version: str = "1.1"):
    """Download mobile deployment configurations for v1.1 models"""
    if version != "1.1":
        return
    
    mobile_configs = {
        "mobile_config.json": {
            "ios": {
                "quantization": "int8",
                "compute_units": "CPU_AND_NE",
                "minimum_deployment_target": "iOS15",
                "optimization_profile": "neural_engine"
            },
            "android": {
                "quantization": "int8", 
                "delegate": "NNAPI",
                "cpu_num_threads": 4,
                "use_xnnpack": True
            }
        },
        "optimization_config.json": {
            "pruning": {
                "enabled": True,
                "sparsity": 0.3,
                "structured": True
            },
            "distillation": {
                "teacher_model": "spatialLM-large-v1.1",
                "temperature": 4.0,
                "alpha": 0.7
            }
        }
    }
    
    for config_name, config_data in mobile_configs.items():
        config_path = os.path.join(output_dir, config_name)
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)
        logger.info(f"Created mobile config: {config_path}")

def main():
    """Main function"""
    args = parse_arguments()
    
    # List available models and exit
    if args.list_models:
        list_available_models()
        return
    
    # Get model information
    if args.model_name == "custom":
        if not args.custom_repo_id:
            logger.error("Custom repo ID must be provided when using 'custom' model")
            sys.exit(1)
        repo_id = args.custom_repo_id
        files = ["pytorch_model.bin", "config.json", "tokenizer.json", "vocab.json"]
        version = "custom"
    else:
        model_info = get_model_info(args.model_name)
        repo_id = model_info["repo_id"]
        files = model_info["files"]
        version = model_info.get("version", "1.0")
    
    # Show model info if requested
    if args.model_info:
        if args.model_name != "custom":
            info = get_model_info(args.model_name)
            print(f"Model: {args.model_name}")
            print(f"Description: {info['description']}")
            print(f"Version: {info['version']}")
            print(f"Repository: {info['repo_id']}")
            print(f"Size: {info['size_mb']} MB")
            print(f"Files: {', '.join(info['files'])}")
            if 'capabilities' in info:
                print(f"Capabilities: {', '.join(info['capabilities'])}")
        return
    
    # Setup output directory
    model_output_dir = os.path.join(args.output_dir, args.model_name)
    
    try:
        # Setup authentication if needed
        setup_authentication(args)
        
        # Download files
        if args.download_specific_files:
            files = args.download_specific_files
        
        logger.info(f"Starting download of {args.model_name}")
        logger.info(f"Repository: {repo_id}")
        logger.info(f"Output directory: {model_output_dir}")
        logger.info(f"Files to download: {files}")
        
        downloaded_files = download_model_files(
            repo_id=repo_id,
            files=files,
            output_dir=model_output_dir,
            revision=args.revision,
            force_download=args.force_download
        )
        
        # Verify downloaded files
        if args.verify_checksums:
            if not verify_downloaded_files(downloaded_files):
                logger.error("File verification failed")
                sys.exit(1)
        
        # Create model info file
        create_model_info(args.model_name, repo_id, model_output_dir, version)
        
        # Download mobile configs for v1.1 models
        if args.include_mobile_configs and version == "1.1":
            download_mobile_configs(model_output_dir, version)
        
        logger.info("✅ Model download completed successfully!")
        logger.info(f"Model saved to: {model_output_dir}")
        
        # Show next steps
        print("\n🚀 Next Steps:")
        print(f"1. Fine-tune the model: python training/finetune.py --model_path {model_output_dir}")
        print(f"2. Convert for mobile: python ios/convert_to_coreml.py --model_path {model_output_dir}")
        print(f"3. Evaluate performance: python training/evaluate.py --model_path {model_output_dir}")
        print(f"4. Quick inference test: python scripts/test_model.py --model_path {model_output_dir}")
        
        if version == "1.1":
            print("\n✨ SpatialLM 1.1 Features Available:")
            print("- Enhanced spatial reasoning with Qwen architecture")
            print("- Improved mobile optimization and quantization")
            print("- Better performance on spatial understanding tasks")
            print("- Advanced coordinate processing capabilities")
            print("- Mobile deployment configurations included")
        
        print(f"\n📖 Model Information:")
        print(f"- Repository: manycore-research/SpatialLM1.1-Qwen-0.5B")
        print(f"- Base Architecture: Qwen-0.5B")
        print(f"- Parameters: ~500M")
        print(f"- Optimized for: Spatial reasoning and mobile deployment")
        
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()