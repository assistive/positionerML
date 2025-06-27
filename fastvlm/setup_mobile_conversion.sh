#!/bin/bash

# FastVLM Mobile Conversion Setup Script
# This script installs all required dependencies for iOS and Android conversion

set -e

echo "🚀 Setting up FastVLM Mobile Conversion Environment"
echo "=================================================="

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "📍 Python version: $python_version"

# Convert version to comparable format (e.g., "3.8" -> 38)
version_number=$(echo $python_version | sed 's/\.//')

if [ "$version_number" -lt 38 ]; then
    echo "❌ Python 3.8+ required. Current version: $python_version"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "mobile_conversion_env" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv mobile_conversion_env
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source mobile_conversion_env/bin/activate

# Upgrade pip
echo "🔧 Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo "📦 Installing core dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy
pip install onnx
pip install onnxruntime

# Install iOS conversion dependencies (CoreMLTools)
echo "📱 Installing iOS conversion dependencies..."
pip install coremltools

# Try to install TensorFlow for Android conversion
echo "🤖 Installing Android conversion dependencies..."
if pip install tensorflow==2.15.0; then
    echo "✅ TensorFlow installed successfully"
    
    # Install additional TensorFlow packages
    pip install tf2onnx
    
    # Try to install onnx-tf (may require specific versions)
    if pip install onnx-tf; then
        echo "✅ ONNX-TensorFlow bridge installed"
    else
        echo "⚠️  ONNX-TensorFlow bridge installation failed - using alternative approach"
    fi
else
    echo "⚠️  TensorFlow installation failed - Android conversion may be limited"
fi

# Install additional utilities
echo "🔧 Installing additional utilities..."
pip install pyyaml
pip install pillow

# Create requirements.txt for reference
echo "📝 Creating requirements.txt..."
cat > requirements_mobile.txt << EOF
# Core dependencies
torch>=2.0.0
torchvision
torchaudio
numpy>=1.21.0
onnx>=1.14.0
onnxruntime>=1.15.0

# iOS conversion
coremltools>=7.0

# Android conversion
tensorflow>=2.13.0,<2.16.0
tf2onnx

# Optional: ONNX-TensorFlow bridge (may require specific versions)
# onnx-tf

# Utilities
pyyaml
pillow
EOF

# Test installations
echo "🧪 Testing installations..."

echo "Testing PyTorch..."
python3 -c "import torch; print(f'✅ PyTorch {torch.__version__}')" || echo "❌ PyTorch test failed"

echo "Testing ONNX..."
python3 -c "import onnx; print(f'✅ ONNX {onnx.__version__}')" || echo "❌ ONNX test failed"

echo "Testing CoreMLTools..."
python3 -c "import coremltools as ct; print(f'✅ CoreMLTools {ct.__version__}')" || echo "❌ CoreMLTools test failed"

echo "Testing TensorFlow..."
python3 -c "import tensorflow as tf; print(f'✅ TensorFlow {tf.__version__}')" || echo "⚠️  TensorFlow not available"

# Create a simple test script
echo "📝 Creating test script..."
cat > test_conversion.py << 'EOF'
#!/usr/bin/env python3
"""
Quick test script for mobile conversion dependencies
"""

def test_dependencies():
    print("🧪 Testing FastVLM Mobile Conversion Dependencies")
    print("=" * 50)
    
    # Test PyTorch
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not available")
        return False
    
    # Test ONNX
    try:
        import onnx
        print(f"✅ ONNX {onnx.__version__}")
    except ImportError:
        print("❌ ONNX not available")
        return False
    
    # Test CoreMLTools
    try:
        import coremltools as ct
        print(f"✅ CoreMLTools {ct.__version__}")
        ios_available = True
    except ImportError:
        print("⚠️  CoreMLTools not available - iOS conversion disabled")
        ios_available = False
    
    # Test TensorFlow
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__}")
        android_available = True
    except ImportError:
        print("⚠️  TensorFlow not available - Android conversion disabled")
        android_available = False
    
    print(f"\n📱 Platform Support:")
    print(f"   iOS (CoreML): {'✅ Available' if ios_available else '❌ Not available'}")
    print(f"   Android (TFLite): {'✅ Available' if android_available else '❌ Not available'}")
    
    if ios_available or android_available:
        print("\n🎉 Ready for mobile conversion!")
        return True
    else:
        print("\n❌ No mobile platforms available")
        return False

if __name__ == "__main__":
    test_dependencies()
EOF

chmod +x test_conversion.py

# Run test
echo "🧪 Running dependency test..."
python3 test_conversion.py

echo ""
echo "✅ Setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Activate the environment: source mobile_conversion_env/bin/activate"
echo "2. Run the converter: python fastvlm_universal_converter.py --platforms ios android"
echo "3. Check output in: ./mobile_models/"
echo ""
echo "🔧 Troubleshooting:"
echo "- If TensorFlow installation fails, try: pip install tensorflow-cpu"
echo "- For M1 Macs, you may need: pip install tensorflow-macos tensorflow-metal"
echo "- If CoreMLTools fails, try: pip install coremltools --no-deps"
echo ""
echo "📚 Documentation:"
echo "- Check conversion_report.json for detailed results"
echo "- Read CONVERSION_SUMMARY.md for deployment instructions"
