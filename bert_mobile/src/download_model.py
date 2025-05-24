# bert_mobile/scripts/download_model.py

#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model_downloader import BERTModelDownloader

def main():
    parser = argparse.ArgumentParser(description='Download BERT models')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                       help='Model name to download')
    parser.add_argument('--output_dir', type=str, default='./models/pretrained',
                       help='Output directory for downloaded model')
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                       help='Path to model configuration file')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download even if model exists')
    parser.add_argument('--list_models', action='store_true',
                       help='List available models')
    parser.add_argument('--compare_models', type=str, nargs='+',
                       help='Compare multiple models')
    parser.add_argument('--recommend', type=str, choices=['mobile', 'accuracy', 'speed'],
                       help='Get model recommendation for use case')
    
    args = parser.parse_args()
    
    try:
        # Initialize downloader
        downloader = BERTModelDownloader(args.config)
        
        if args.list_models:
            print("Available BERT models:")
            for model in downloader.list_available_models():
                info = downloader.get_model_info(model)
                print(f"  - {model}: {info.get('description', 'No description')}")
                print(f"    Size: {info.get('size_mb', 'Unknown')}MB, "
                      f"Layers: {info.get('num_layers', 'Unknown')}")
            return
        
        if args.recommend:
            recommended = downloader.get_recommended_model(args.recommend)
            print(f"Recommended model for {args.recommend}: {recommended}")
            info = downloader.get_model_info(recommended)
            print(f"Description: {info.get('description', 'No description')}")
            return
        
        if args.compare_models:
            comparison = downloader.compare_models(args.compare_models)
            print("Model Comparison:")
            for model, info in comparison['models'].items():
                print(f"\n{model}:")
                for key, value in info.items():
                    print(f"  {key}: {value}")
            
            if 'summary' in comparison:
                print(f"\nSummary:")
                for key, value in comparison['summary'].items():
                    print(f"  {key}: {value}")
            return
        
        # Download model
        print(f"Downloading BERT model: {args.model_name}")
        
        # Update config to use custom output directory
        downloader.config['model']['cache_dir'] = args.output_dir
        downloader.cache_dir = Path(args.output_dir)
        downloader.cache_dir.mkdir(parents=True, exist_ok=True)
        
        model_path, tokenizer_path = downloader.download_model(
            args.model_name,
            force_download=args.force
        )
        
        print(f"Model downloaded successfully!")
        print(f"Model path: {model_path}")
        print(f"Tokenizer path: {tokenizer_path}")
        
        # Get model info
        info = downloader.get_model_info(args.model_name)
        print(f"\nModel Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Check compatibility
        compatibility = downloader.check_model_compatibility(model_path)
        print(f"\nMobile Compatibility:")
        for key, value in compatibility.items():
            status = "✓" if value else "✗"
            print(f"  {status} {key}: {value}")
        
        # Load and test model
        print(f"\nTesting model loading...")
        model, tokenizer, config = downloader.load_model_and_tokenizer(model_path)
        
        # Test tokenization
        test_text = "Hello, this is a test sentence for BERT tokenization."
        tokens = tokenizer.tokenize(test_text)
        print(f"Test tokenization: {test_text}")
        print(f"Tokens: {tokens[:10]}..." if len(tokens) > 10 else f"Tokens: {tokens}")
        
        print(f"\nModel ready for training/fine-tuning!")
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

