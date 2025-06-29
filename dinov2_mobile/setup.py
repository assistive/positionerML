#!/usr/bin/env python3
"""
DINOv2 Mobile Deployment Setup Script
Installs all required dependencies for mobile conversion
"""
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def check_dependency(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        # Handle special cases where package name != import name
        import_map = {
            "pillow": "PIL",
            "pyyaml": "yaml",
            "opencv-python": "cv2",
            "scikit-learn": "sklearn",
            "tensorflow-addons": "tensorflow_addons",
            "onnx-tf": "onnx_tf"
        }
        import_name = import_map.get(package_name.lower(), package_name)
    
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False

def main():
    print("🚀 Setting up DINOv2 Mobile Deployment Environment")
    print("="*50)
    
    # Core dependencies
    core_deps = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("pillow", "Pillow"),
        ("pyyaml", "PyYAML")
    ]
    
    # Mobile conversion dependencies
    mobile_deps = [
        ("coremltools", "CoreML Tools (iOS)"),
        ("tensorflow", "TensorFlow (Android)"),
        ("onnx", "ONNX"),
        ("onnxruntime", "ONNX Runtime"),
    ]
    
    # Optional dependencies for enhanced conversion
    optional_deps = [
        ("onnx-tf", "ONNX-TensorFlow Bridge")
    ]
    
    print("📦 Checking core dependencies...")
    missing_core = []
    for pkg, name in core_deps:
        if check_dependency(pkg):
            print(f"✅ {name} is installed")
        else:
            print(f"❌ {name} is missing")
            missing_core.append(pkg)
    
    print("\n📱 Checking mobile conversion dependencies...")
    missing_mobile = []
    for pkg, name in mobile_deps:
        if check_dependency(pkg):
            print(f"✅ {name} is installed")
        else:
            print(f"❌ {name} is missing")
            missing_mobile.append(pkg)
    
    print("\n🔧 Checking optional dependencies...")
    missing_optional = []
    for pkg, name in optional_deps:
        if check_dependency(pkg.replace('-', '_')):
            print(f"✅ {name} is installed")
        else:
            print(f"⚠️  {name} is missing (optional)")
            missing_optional.append(pkg)
    
    # Install missing dependencies
    all_missing = missing_core + missing_mobile + missing_optional
    
    if all_missing:
        print(f"\n🔄 Installing {len(all_missing)} missing packages...")
        
        # Install core dependencies first
        if missing_core:
            cmd = f"pip install {' '.join(missing_core)}"
            if not run_command(cmd, "Installing core dependencies"):
                print("❌ Failed to install core dependencies. Please install manually.")
                return False
        
        # Install mobile dependencies
        if missing_mobile:
            # Special handling for TensorFlow on M1 Macs
            if "tensorflow" in missing_mobile and sys.platform == "darwin":
                print("🍎 Detected macOS - installing TensorFlow for Apple Silicon...")
                cmd = "pip install tensorflow-macos tensorflow-metal"
                if not run_command(cmd, "Installing TensorFlow for macOS"):
                    # Fallback to regular TensorFlow
                    cmd = "pip install tensorflow"
                    run_command(cmd, "Installing regular TensorFlow")
                missing_mobile.remove("tensorflow")
            
            if missing_mobile:
                cmd = f"pip install {' '.join(missing_mobile)}"
                run_command(cmd, "Installing mobile conversion dependencies")
        
        # Install optional dependencies
        if missing_optional:
            for pkg in missing_optional:
                run_command(f"pip install {pkg}", f"Installing {pkg}")
    
    print("\n🎯 Verifying installation...")
    
    # Verify critical dependencies
    critical_imports = [
        ("torch", "PyTorch"),
        ("coremltools", "CoreML Tools"),
        ("tensorflow", "TensorFlow")
    ]
    
    all_good = True
    for pkg, name in critical_imports:
        if check_dependency(pkg):
            print(f"✅ {name} verified")
        else:
            print(f"❌ {name} verification failed")
            all_good = False
    
    if all_good:
        print("\n🎉 Setup completed successfully!")
        print("You can now run the conversion script:")
        print("  python scripts/convert/convert_dinov2_enhanced.py --model dinov2_vits14")
    else:
        print("\n⚠️  Setup completed with some issues.")
        print("Please manually install missing dependencies:")
        print("  pip install torch torchvision coremltools tensorflow")
        
    return all_good

if __name__ == "__main__":
    main()
