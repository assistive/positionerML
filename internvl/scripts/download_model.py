# internvl/scripts/download_model.py

#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model_downloader import ModelDownloader

def main():
    parser = argparse.ArgumentParser(description='Download InternVL models')
    parser.add_argument('--model', type=str, default='internvl2-2b',
                       help='Model to download (default: internvl2-2b)')
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                       help='Path to model configuration file')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download even if model exists')
    parser.add_argument('--list-models', action='store_true',
                       help='List available models')
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = ModelDownloader(args.config)
    
    if args.list_models:
        print("Available InternVL models:")
        for model in downloader.list_available_models():
            print(f"  - {model}")
        return
    
    try:
        # Download model
        model_path, tokenizer_path = downloader.download_model(
            model_name=None if args.model == 'internvl2-2b' else args.model,
            force_download=args.force
        )
        
        print(f"Model downloaded successfully!")
        print(f"Model path: {model_path}")
        print(f"Tokenizer path: {tokenizer_path}")
        
        # Get model info
        info = downloader.get_model_info(model_path)
        print(f"\nModel Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Check compatibility
        compatibility = downloader.check_model_compatibility(model_path)
        print(f"\nCompatibility:")
        for key, value in compatibility.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
