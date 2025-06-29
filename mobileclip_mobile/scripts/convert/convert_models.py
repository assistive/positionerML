#!/usr/bin/env python3
"""
MobileCLIP Mobile Conversion Script
Converts MobileCLIP models to mobile-optimized formats
"""
import argparse
import logging
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from mobile_converter.converter import MobileCLIPMobileConverter

def main():
    parser = argparse.ArgumentParser(description="Convert MobileCLIP models for mobile deployment")
    parser.add_argument("--model", choices=["mobileclip_s0", "mobileclip_s1", "mobileclip_s2", "mobileclip_b", "mobileclip_blt"], 
                       default="mobileclip_s0", help="MobileCLIP model variant")
    parser.add_argument("--platforms", nargs="+", choices=["ios", "android"], 
                       default=["ios", "android"], help="Target platforms")
    parser.add_argument("--model-path", help="Path to pretrained model file")
    parser.add_argument("--output", default="./models/converted", help="Output directory")
    parser.add_argument("--config", default="./config/mobileclip_config.yaml", help="Config file")
    parser.add_argument("--validate", action="store_true", help="Validate conversion accuracy")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create converter
    converter = MobileCLIPMobileConverter(config)
    
    # Load model
    print(f"🔄 Loading {args.model}...")
    try:
        converter.load_pytorch_model(args.model, args.model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Convert for each platform
    for platform in args.platforms:
        print(f"\n📱 Converting for {platform.upper()}...")
        
        try:
            if platform == "ios":
                model_path = converter.convert_to_coreml(str(output_path), args.model)
                results[platform] = model_path
                
            elif platform == "android":
                model_path = converter.convert_to_tflite(str(output_path), args.model)
                results[platform] = model_path
                
            print(f"✅ {platform.upper()} conversion completed: {model_path}")
            
        except Exception as e:
            print(f"❌ {platform.upper()} conversion failed: {e}")
            results[platform] = f"ERROR: {str(e)}"
    
    # Print summary
    print("\n" + "="*50)
    print("CONVERSION SUMMARY")
    print("="*50)
    
    for platform, result in results.items():
        if not result.startswith("ERROR"):
            print(f"✅ {platform.upper()}: {result}")
            # Get model size info
            try:
                if platform == "ios":
                    # CoreML packages are directories
                    print(f"   📊 CoreML package created")
                elif platform == "android":
                    size_mb = Path(result).stat().st_size / (1024 * 1024)
                    print(f"   📊 Size: {size_mb:.1f} MB")
            except:
                pass
        else:
            print(f"❌ {platform.upper()}: {result}")
    
    print("="*50)

if __name__ == "__main__":
    main()
