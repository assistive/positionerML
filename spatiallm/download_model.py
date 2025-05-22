#!/usr/bin/env python3
"""
download_model.py

Script to download pre-trained spatialLM models from HuggingFace or other repositories.
Handles authentication, model selection, and verification of downloaded files.
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

# Default models info
DEFAULT_MODELS = {
    "spatialLM-base": {
        "repo_id": "spatialLM/spatialLM-base",
        "files": ["pytorch_model.bin", "config.json", "tokenizer.json", "vocab.json"],
        "description": "Base model (220M parameters)",
        "size_mb": 850
    },
    "spatialLM-small": {
        "repo_id": "spatialLM/spatialLM-small",
        "files": ["pytorch_model.bin", "config.json", "tokenizer.json", "vocab.json"],
        "description": "Small model (70M parameters)",
        "size_mb": 280
    },
    "spatialLM-large": {
        "repo_id": "spatialLM/spatialLM-large",
        "files": ["pytorch_model.bin", "config.json", "tokenizer.json", "vocab.json"],
        "description": "Large model (770M parameters)",
        "size_mb": 3100
    }
}

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Download pre-trained spatialLM models")
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="spatialLM-base",
        choices=list(DEFAULT_MODELS.keys()) + ["custom"],
        help="Name of the model to download"
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
        help="Force download even if files already exist locally"
    )
    
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="List available models and exit"
    )
    
    parser.add_argument(
        "--verify_downloads",
        action="store_true",
        help="Verify downloaded files integrity"
    )
    
    return parser.parse_args()

def list_available_models():
    """Print information about available models"""
    logger.info("Available spatialLM models:")
    logger.info("-" * 80)
    
    format_str = "{:<15} {:<12} {:<40}"
    logger.info(format_str.format("Model", "Size", "Description"))
    logger.info("-" * 80)
    
    for model_name, model_info in DEFAULT_MODELS.items():
        size_str = f"{model_info['size_mb']} MB"
        logger.info(format_str.format(model_name, size_str, model_info['description']))

def get_auth_token(args):
    """Get HuggingFace authentication token"""
    if args.auth_token:
        return args.auth_token
    
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if not token and args.use_auth_token:
        logger.info("No token provided. Please login to HuggingFace:")
        login()
        return True
    
    return token

def create_output_directory(output_dir, model_name):
    """Create output directory for the model"""
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def download_model_files(repo_id, output_dir, files=None, auth_token=None, revision="main", force_download=False):
    """Download model files from HuggingFace Hub"""
    try:
        if files:
            # Download specific files
            for file in files:
                logger.info(f"Downloading {file}...")
                hf_hub_download(
                    repo_id=repo_id,
                    filename=file,
                    local_dir=output_dir,
                    token=auth_token,
                    revision=revision,
                    force_download=force_download
                )
        else:
            # Download entire repository
            logger.info(f"Downloading all files from {repo_id}...")
            snapshot_download(
                repo_id=repo_id,
                local_dir=output_dir,
                token=auth_token,
                revision=revision,
                force_download=force_download
            )
        return True
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return False

def verify_downloads(output_dir, files=None):
    """Verify the integrity of downloaded files"""
    if not files:
        files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
    
    all_valid = True
    for file in files:
        file_path = os.path.join(output_dir, file)
        if os.path.exists(file_path):
            # Check file size
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            logger.info(f"File {file}: {size_mb:.2f} MB")
            
            # Calculate file hash
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            file_hash = sha256_hash.hexdigest()
            logger.info(f"SHA256: {file_hash}")
        else:
            logger.error(f"File {file} not found")
            all_valid = False
    
    return all_valid

def create_model_info(model_name, repo_id, output_dir):
    """Create a model_info.json file with details about the downloaded model"""
    info = {
        "model_name": model_name,
        "repo_id": repo_id,
        "download_date": datetime.datetime.now().isoformat(),
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
            logger.error("Custom repo ID must be provided when using --model_name=custom")
            return
        
        repo_id = args.custom_repo_id
        files = None  # Download all files for custom models
        model_name = repo_id.split("/")[-1]
    else:
        if args.model_name not in DEFAULT_MODELS:
            logger.error(f"Unknown model: {args.model_name}")
            list_available_models()
            return
        
        model_info = DEFAULT_MODELS[args.model_name]
        repo_id = model_info["repo_id"]
        files = model_info["files"]
        model_name = args.model_name
    
    # Get authentication token if needed
    auth_token = get_auth_token(args) if args.use_auth_token else None
    
    # Create output directory
    model_dir = create_output_directory(args.output_dir, model_name)
    logger.info(f"Downloading {model_name} to {model_dir}")
    
    # Download model files
    success = download_model_files(
        repo_id=repo_id,
        output_dir=model_dir,
        files=files,
        auth_token=auth_token,
        revision=args.revision,
        force_download=args.force_download
    )
    
    if not success:
        logger.error("Failed to download model")
        return
    
    # Verify downloads if requested
    if args.verify_downloads:
        logger.info("Verifying downloaded files...")
        if verify_downloads(model_dir, files):
            logger.info("All files verified successfully")
        else:
            logger.warning("Some files could not be verified")
    
    # Create model info file
    create_model_info(model_name, repo_id, model_dir)
    
    logger.info(f"Model {model_name} downloaded successfully to {model_dir}")

if __name__ == "__main__":
    # Add datetime import for model_info creation
    import datetime
    main()
