#!/usr/bin/env python3
"""
Download Qwen 2.5-VL models from Hugging Face.
"""

import argparse
import logging
from pathlib import Path
import sys
import os

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AVAILABLE_MODELS = {
    "3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "7b": "Qwen/Qwen2.5-VL-7B-Instruct", 
    "32b": "Qwen/Qwen2.5-VL-32B-Instruct",
    "72b": "Qwen/Qwen2.5-VL-72B-Instruct"
}

def download_model(model_size: str, output_dir: str, force: bool = False):
    """Download Qwen 2.5-VL model."""
    
    if model_size not in AVAILABLE_MODELS:
        raise ValueError(f"Model size must be one of: {list(AVAILABLE_MODELS.keys())}")
    
    model_id = AVAILABLE_MODELS[model_size]
    output_path = Path(output_dir) / model_size
    
    if output_path.exists() and not force:
        logger.info(f"Model already exists at {output_path}. Use --force to overwrite.")
        return
    
    logger.info(f"Downloading {model_id} to {output_path}")
    
    try:
        # Download model files
        snapshot_download(
            repo_id=model_id,
            local_dir=str(output_path),
            local_dir_use_symlinks=False
        )
        
        logger.info(f"Successfully downloaded {model_id}")
        
        # Verify the download by loading the model briefly
        logger.info("Verifying download...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(output_path),
            torch_dtype="auto",
            device_map="cpu"  # Load on CPU for verification
        )
        processor = AutoProcessor.from_pretrained(str(output_path))
        
        logger.info("Download verification successful!")
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {total_params:,}")
        
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        # Cleanup partial download
        if output_path.exists():
            import shutil
            shutil.rmtree(output_path)
        raise

def main():
    parser = argparse.ArgumentParser(description="Download Qwen 2.5-VL models")
    parser.add_argument(
        "model_size",
        choices=list(AVAILABLE_MODELS.keys()),
        help="Model size to download"
    )
    parser.add_argument(
        "--output-dir",
        default="models/pretrained",
        help="Output directory for downloaded models"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model exists"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        download_model(args.model_size, args.output_dir, args.force)
        logger.info("Download completed successfully!")
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
