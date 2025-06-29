#!/bin/bash

# MobileCLIP Mobile Deployment - Complete Setup Script
# This script sets up the complete environment for MobileCLIP mobile deployment

set -e

echo "🚀 Setting up MobileCLIP Mobile Deployment Environment..."
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Anaconda or Miniconda first."
    echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment
echo "📦 Creating conda environment 'mobileclip_mobile'..."
conda create -n mobileclip_mobile python=3.10 -y

echo "🔄 Activating environment..."
eval "$(conda shell.bash hook)"
conda activate mobileclip_mobile

echo "📥 Installing core dependencies..."
# Install PyTorch
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Install basic requirements
pip install --upgrade pip
pip install numpy pillow pyyaml tqdm psutil

echo "📥 Installing MobileCLIP from Apple's repository..."
# Install MobileCLIP directly from GitHub
pip install git+https://github.com/apple/ml-mobileclip.git

echo "📥 Installing mobile conversion dependencies..."
# Install CoreML Tools (macOS only)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 Installing CoreML Tools (macOS detected)..."
    pip install coremltools
else
    echo "⚠️  CoreML Tools skipped (not macOS - iOS conversion will not work)"
fi

# Install TensorFlow for Android conversion
pip install tensorflow>=2.13.0

# Install ONNX for conversion pipeline
pip install onnx onnxruntime

# Install additional utilities
pip install huggingface_hub transformers

echo "✅ Installation complete!"
echo ""
echo "🧪 Testing MobileCLIP installation..."

# Test MobileCLIP installation
python -c "
import torch
try:
    import mobileclip
    print('✅ MobileCLIP imported successfully')
    
    # Test model creation
    model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained=None)
    print('✅ MobileCLIP model creation works')
    
    tokenizer = mobileclip.get_tokenizer('mobileclip_s0')
    print('✅ MobileCLIP tokenizer works')
    
except Exception as e:
    print(f'❌ MobileCLIP test failed: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo "✅ MobileCLIP installation verified!"
else
    echo "❌ MobileCLIP installation verification failed!"
    exit 1
fi

echo ""
echo "🎯 Next steps:"
echo "1. conda activate mobileclip_mobile"
echo "2. python scripts/download/download_models.py --models mobileclip_s0"
echo "3. python scripts/convert/convert_models.py --model mobileclip_s0 --platforms ios android"
echo "4. python scripts/deploy/deploy_mobile.py --zip"
echo ""
echo "📚 For detailed usage, see README.md and docs/"
