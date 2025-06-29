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
