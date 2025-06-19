#!/usr/bin/env python3
"""
Run the Qwen 2.5-VL service.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qwen_vl.service import QwenVLService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run Qwen 2.5-VL service")
    parser.add_argument(
        "--config",
        default="config/service_config.yaml",
        help="Service configuration file"
    )
    parser.add_argument(
        "--model-variant",
        default="qwen-2.5-vl-7b",
        help="Model variant to load"
    )
    parser.add_argument(
        "--host",
        help="Host to bind to (overrides config)"
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port to bind to (overrides config)"
    )
    parser.add_argument(
        "--auto-load-model",
        action="store_true",
        help="Automatically load model on startup"
    )
    
    args = parser.parse_args()
    
    try:
        # Create service
        service = QwenVLService(config_path=args.config)
        
        # Load model if requested
        if args.auto_load_model:
            logger.info(f"Loading model: {args.model_variant}")
            service.model_manager.load_model(args.model_variant)
        
        # Run service
        logger.info("Starting Qwen 2.5-VL service...")
        service.run(host=args.host, port=args.port)
        
    except Exception as e:
        logger.error(f"Service failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
