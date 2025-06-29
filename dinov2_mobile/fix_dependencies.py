#!/usr/bin/env python3
"""
Quick fix for common DINOv2 mobile conversion dependency issues
"""
import subprocess
import sys

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("🔧 Fixing DINOv2 Mobile Conversion Dependencies")
    print("=" * 50)
    
    # Common problematic packages and their fixes
    fixes = [
        ("tensorflow-addons", "TensorFlow Addons (required by onnx-tf)"),
        ("onnx-tf", "ONNX to TensorFlow converter"),
        ("protobuf==3.20.3", "Compatible protobuf version"),
    ]
    
    for package, description in fixes:
        print(f"📦 Installing {description}...")
        if install_package(package):
            print(f"✅ {package} installed successfully")
        else:
            print(f"❌ Failed to install {package}")
    
    # Additional fixes for common issues
    print("\n🔄 Applying compatibility fixes...")
    
    # Fix potential protobuf version conflicts
    compatibility_packages = [
        "protobuf==3.20.3",  # Known working version
        "onnx==1.14.1",      # Compatible ONNX version
    ]
    
    for package in compatibility_packages:
        print(f"🔧 Installing {package}...")
        install_package(package)
    
    print("\n✅ Dependency fixes applied!")
    print("Now try running the conversion again:")
    print("  python scripts/convert/convert_dinov2_enhanced.py --model dinov2_vits14")

if __name__ == "__main__":
    main()
