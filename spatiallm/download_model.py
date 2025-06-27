#!/usr/bin/env python3
"""
download_model.py

Script to download pre-trained spatialLM v1.1 models from HuggingFace.
This should be placed in: spatiallm/download_model.py (root of spatiallm folder)

Updated for the actual SpatialLM 1.1 repository:
https://huggingface.co/manycore-research/SpatialLM1.1-Qwen-0.5B
"""

import os
import argparse
import logging
import sys
import hashlib
import json
import datetime
from pathlib import Path
from huggingface_hub import snapshot_download, login
from typing import Optional, List, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("download_model")

# Updated models info for v1.1 - Using actual Hugging Face repository
DEFAULT_MODELS = {
    "spatiallm-1.1-qwen-0.5b": {
        "repo_id": "manycore-research/SpatialLM1.1-Qwen-0.5B",
        "files": ["pytorch_model.bin", "config.json", "tokenizer.json", "tokenizer_config.json"],
        "description": "SpatialLM 1.1 based on Qwen-0.5B (500M parameters) - Enhanced spatial reasoning",
        "size_mb": 1900,
        "version": "1.1",
        "capabilities": ["enhanced_spatial_reasoning", "qwen_architecture", "mobile_optimized", "quantization_ready"],
        "base_architecture": "qwen"
    },
    # Alias for easier access
    "spatiallm-base": {
        "repo_id": "manycore-research/SpatialLM1.1-Qwen-0.5B",
        "files": ["pytorch_model.bin", "config.json", "tokenizer.json", "tokenizer_config.json"],
        "description": "SpatialLM 1.1 base model (500M parameters) - Enhanced spatial reasoning",
        "size_mb": 1900,
        "version": "1.1",
        "capabilities": ["enhanced_spatial_reasoning", "qwen_architecture", "mobile_optimized", "quantization_ready"],
        "base_architecture": "qwen"
    }
}

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Download pre-trained spatialLM models")
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="spatiallm-1.1-qwen-0.5b",
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
        "--list_models",
        action="store_true",
        help="List all available models and exit"
    )
    
    parser.add_argument(
        "--model_info",
        action="store_true",
        help="Show detailed information about the selected model"
    )
    
    return parser.parse_args()

def list_available_models():
    """List all available models with their details"""
    print("Available spatialLM models:")
    print("=" * 80)
    
    for name, info in DEFAULT_MODELS.items():
        print(f"\n📋 {name}:")
        print(f"    Description: {info['description']}")
        print(f"    Size: {info['size_mb']} MB")
        print(f"    Version: {info['version']}")
        print(f"    Architecture: {info.get('base_architecture', 'transformer')}")
        print(f"    Capabilities: {', '.join(info['capabilities'])}")
        print(f"    Repository: {info['repo_id']}")

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

def download_model_from_hub(repo_id: str, output_dir: str, revision: str = "main", force_download: bool = False):
    """Download model from HuggingFace Hub"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"Downloading model from {repo_id}...")
        
        # Download the entire model repository
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=str(output_path.parent),
            force_download=force_download,
            local_files_only=False
        )
        
        # Copy to our desired output directory
        import shutil
        if Path(downloaded_path) != output_path:
            if output_path.exists():
                shutil.rmtree(output_path)
            shutil.copytree(downloaded_path, output_path)
        
        logger.info(f"✓ Model downloaded successfully to: {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        raise

def verify_downloaded_files(model_path: str, expected_files: List[str]) -> bool:
    """Verify that all expected files were downloaded successfully"""
    logger.info("Verifying downloaded files...")
    
    all_valid = True
    for file in expected_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            logger.info(f"✓ {file}: {size_mb:.2f} MB")
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
        "base_architecture": "qwen",
        "files": []
    }
    
    # Get list of files and their sizes/hashes
    for root, _, files in os.walk(output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, output_dir)
            
            # Calculate file hash for verification
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
    info_path = os.path.join(output_dir, "model_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"Created model info file: {info_path}")

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
        expected_files = ["pytorch_model.bin", "config.json", "tokenizer.json"]
        version = "custom"
        model_info = {"description": f"Custom model from {repo_id}"}
    else:
        model_info = get_model_info(args.model_name)
        repo_id = model_info["repo_id"]
        expected_files = model_info["files"]
        version = model_info.get("version", "1.0")
    
    # Show model info if requested
    if args.model_info:
        print(f"Model: {args.model_name}")
        print(f"Description: {model_info['description']}")
        print(f"Version: {version}")
        print(f"Repository: {repo_id}")
        if 'size_mb' in model_info:
            print(f"Size: {model_info['size_mb']} MB")
        if 'capabilities' in model_info:
            print(f"Capabilities: {', '.join(model_info['capabilities'])}")
        return
    
    # Setup output directory
    model_output_dir = os.path.join(args.output_dir, args.model_name)
    
    try:
        # Setup authentication if needed
        setup_authentication(args)
        
        logger.info(f"Starting download of {args.model_name}")
        logger.info(f"Repository: {repo_id}")
        logger.info(f"Output directory: {model_output_dir}")
        
        # Download the model
        downloaded_path = download_model_from_hub(
            repo_id=repo_id,
            output_dir=model_output_dir,
            revision=args.revision,
            force_download=args.force_download
        )
        
        # Verify downloaded files
        if not verify_downloaded_files(downloaded_path, expected_files):
            logger.warning("Some files may be missing, but continuing...")
        
        # Create model info file
        create_model_info(args.model_name, repo_id, downloaded_path, version)
        
        logger.info("✅ Model download completed successfully!")
        logger.info(f"Model saved to: {downloaded_path}")
        
        # Show next steps
        print("\n🚀 Next Steps:")
        print(f"1. Test the model: python test_model.py --model_path {downloaded_path}")
        print(f"2. Convert for iOS: python convert_to_coreml.py --model_path {downloaded_path}")
        print(f"3. Convert for Android: python convert_to_tflite.py --model_path {downloaded_path}")
        
        if version == "1.1":
            print("\n✨ SpatialLM 1.1 Features Available:")
            print("- Enhanced spatial reasoning with Qwen architecture")
            print("- Improved mobile optimization and quantization")
            print("- Better performance on spatial understanding tasks")
            print("- Advanced coordinate processing capabilities")
        
        print(f"\n📖 Model Information:")
        print(f"- Repository: {repo_id}")
        print(f"- Base Architecture: Qwen-0.5B")
        print(f"- Parameters: ~500M")
        print(f"- Optimized for: Spatial reasoning and mobile deployment")
        
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()