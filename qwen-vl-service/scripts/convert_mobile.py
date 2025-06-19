#!/usr/bin/env python3
"""
Convert Qwen 2.5-VL model for mobile deployment.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qwen_vl.model_manager import QwenVLModelManager
from qwen_vl.mobile_converter import QwenVLMobileConverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Convert Qwen 2.5-VL for mobile")
    parser.add_argument(
        "model_variant",
        help="Model variant to convert"
    )
    parser.add_argument(
        "--platform",
        choices=["ios", "android", "both"],
        default="both",
        help="Target platform"
    )
    parser.add_argument(
        "--quantization-bits",
        type=int,
        choices=[4, 8, 16],
        default=8,
        help="Quantization bits"
    )
    parser.add_argument(
        "--pruning-sparsity",
        type=float,
        default=0.3,
        help="Pruning sparsity (0.0-1.0)"
    )
    parser.add_argument(
        "--output-dir",
        default="models/mobile",
        help="Output directory for mobile models"
    )
    
    args = parser.parse_args()
    
    try:
        # Load model
        logger.info(f"Loading model: {args.model_variant}")
        model_manager = QwenVLModelManager()
        model_manager.load_model(args.model_variant)
        
        # Create converter
        converter = QwenVLMobileConverter()
        
        # Convert for each platform
        platforms = ["ios", "android"] if args.platform == "both" else [args.platform]
        
        for platform in platforms:
            logger.info(f"Converting for {platform}")
            
            results = converter.optimize_for_mobile(
                model=model_manager.model,
                variant=args.model_variant,
                target_platform=platform
            )
            
            logger.info(f"Conversion results for {platform}: {results}")
        
        logger.info("Mobile conversion completed successfully!")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
