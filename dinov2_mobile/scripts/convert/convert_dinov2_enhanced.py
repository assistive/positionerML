#!/usr/bin/env python3
"""
Enhanced DINOv2 Mobile Conversion Script with Error Handling
"""
import argparse
import sys
import logging
from pathlib import Path
import yaml
import subprocess

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    
    # Check dependencies with correct import names
    deps_to_check = [
        ("torch", "torch"),
        ("coremltools", "coremltools"), 
        ("tensorflow", "tensorflow"),
        ("onnx", "onnx"),
        ("pillow", "PIL"),
        ("pyyaml", "yaml"),
        ("numpy", "numpy")
    ]
    
    for package_name, import_name in deps_to_check:
        try:
            __import__(import_name)
            print(f"✅ {package_name} is available")
        except ImportError:
            print(f"❌ {package_name} is missing")
            missing.append(package_name)
    
    return missing

def install_missing_deps(missing_deps):
    """Attempt to install missing dependencies."""
    if not missing_deps:
        return True
        
    print(f"\n🔄 Attempting to install missing dependencies: {', '.join(missing_deps)}")
    
    try:
        cmd = f"pip install {' '.join(missing_deps)}"
        subprocess.run(cmd, shell=True, check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies automatically")
        print("Please run the setup script first:")
        print("  python setup.py")
        return False

def main():
    parser = argparse.ArgumentParser(description="Enhanced DINOv2 mobile conversion")
    parser.add_argument("--model", choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"], 
                       default="dinov2_vits14", help="DINOv2 model variant")
    parser.add_argument("--platforms", nargs="+", choices=["ios", "android"], 
                       default=["ios", "android"], help="Target platforms")
    parser.add_argument("--output", default="./models/converted", help="Output directory")
    parser.add_argument("--config", default="./config/dinov2_config.yaml", help="Config file")
    parser.add_argument("--validate", action="store_true", help="Validate conversion accuracy")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--cpu-only", action="store_true", help="Use CPU-only conversion (safer)")
    parser.add_argument("--auto-install", action="store_true", help="Auto-install missing dependencies")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🔍 Checking dependencies...")
    missing_deps = check_dependencies()
    
    if missing_deps:
        if args.auto_install:
            if not install_missing_deps(missing_deps):
                sys.exit(1)
        else:
            print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
            print("Run with --auto-install to install them automatically, or run:")
            print("  python setup.py")
            sys.exit(1)
    
    # Import after dependency check
    from mobile_converter.converter import DINOv2MobileConverter
    
    # Load configuration
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"⚠️  Config file {args.config} not found, using defaults")
        config = {
            "quantization": {"enabled": True},
            "optimization": {"cpu_only": args.cpu_only}
        }
    
    # Create converter
    converter = DINOv2MobileConverter(config)
    
    # Load model
    print(f"🔄 Loading {args.model}...")
    try:
        converter.load_pytorch_model(args.model)
        print(f"✅ Model {args.model} loaded successfully")
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
                print("🍎 Starting iOS (CoreML) conversion...")
                print("   This may take several minutes...")
                model_path = converter.convert_to_coreml(str(output_path))
                results[platform] = model_path
                
            elif platform == "android":
                print("🤖 Starting Android (TensorFlow Lite) conversion...")
                print("   This may take several minutes...")
                model_path = converter.convert_to_tflite(str(output_path))
                results[platform] = model_path
                
            print(f"✅ {platform.upper()} conversion completed: {model_path}")
            
            # Get file size
            try:
                size_mb = Path(model_path).stat().st_size / (1024 * 1024)
                print(f"   📊 Model size: {size_mb:.1f} MB")
            except:
                pass
                
        except Exception as e:
            print(f"❌ {platform.upper()} conversion failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            results[platform] = f"ERROR: {str(e)}"
    
    # Print summary
    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)
    
    success_count = 0
    for platform, result in results.items():
        if not result.startswith("ERROR"):
            print(f"✅ {platform.upper()}: SUCCESS")
            print(f"   📁 {result}")
            success_count += 1
        else:
            print(f"❌ {platform.upper()}: FAILED")
            print(f"   ⚠️  {result}")
    
    print("="*60)
    
    if success_count > 0:
        print(f"🎉 {success_count}/{len(args.platforms)} conversions completed successfully!")
        print("\nNext steps:")
        print("1. Create deployment packages:")
        print("   python scripts/deploy/deploy_mobile.py")
        print("2. See deployment guides in the generated packages")
    else:
        print("❌ All conversions failed. Check the error messages above.")
        print("\nTroubleshooting:")
        print("1. Try running with --cpu-only flag")
        print("2. Make sure all dependencies are installed: python setup.py")
        print("3. Check the verbose output with --verbose flag")

if __name__ == "__main__":
    main()
