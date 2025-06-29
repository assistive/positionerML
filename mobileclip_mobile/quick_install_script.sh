#!/bin/bash

# Definitive MobileCLIP Installation Fix
# This script handles all the edge cases and ensures mobileclip package is properly installed

echo "🔧 Definitive MobileCLIP Installation Fix..."

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check basic requirements
if ! command_exists python; then
    echo "❌ Python not found. Please install Python 3.8+ first."
    exit 1
fi

if ! command_exists git; then
    echo "❌ Git not found. Please install Git first."
    exit 1
fi

echo "🧹 Cleaning up any previous MobileCLIP installations..."
pip uninstall mobileclip -y 2>/dev/null || true

echo "📦 Installing ALL dependencies first..."
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install timm transformers tokenizers huggingface_hub
pip install numpy pillow pyyaml tqdm psutil requests
pip install ftfy regex

# Platform-specific dependencies
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 macOS detected - installing CoreML Tools..."
    pip install coremltools
fi

pip install tensorflow onnx onnxruntime

# Use current directory for MobileCLIP installation
echo "📁 Working in current directory: $(pwd)"

# Clean up any existing installation
if [ -d "ml-mobileclip" ]; then
    echo "🗑️  Removing existing ml-mobileclip directory..."
    rm -rf ml-mobileclip
fi

echo "📥 Cloning MobileCLIP repository..."
git clone https://github.com/apple/ml-mobileclip.git
cd ml-mobileclip

echo "🔍 Checking repository structure..."
if [ ! -d "mobileclip" ]; then
    echo "❌ ERROR: 'mobileclip' directory not found in repository!"
    echo "Repository contents:"
    ls -la
    echo ""
    echo "This suggests the repository structure has changed."
    echo "Let's try alternative installation methods..."
    
    # Try installing requirements first
    if [ -f "requirements.txt" ]; then
        echo "📦 Installing from requirements.txt..."
        pip install -r requirements.txt
    fi
    
    # Try the installation anyway
    echo "📦 Attempting installation anyway..."
    pip install -e . || {
        echo "❌ Standard installation failed."
        echo ""
        echo "🔄 Trying alternative: OpenCLIP with MobileCLIP support..."
        cd ..
        pip install open_clip_torch
        
        echo "🧪 Testing OpenCLIP with MobileCLIP..."
        python -c "
import open_clip
try:
    model, _, preprocess = open_clip.create_model_and_transforms('MobileCLIP-S2', pretrained='datacompdr')
    tokenizer = open_clip.get_tokenizer('MobileCLIP-S2')
    print('✅ MobileCLIP available through OpenCLIP!')
    print('   Use open_clip instead of mobileclip in your scripts.')
    print('   Example: open_clip.create_model_and_transforms(\"MobileCLIP-S2\", pretrained=\"datacompdr\")')
except Exception as e:
    print(f'❌ OpenCLIP MobileCLIP failed: {e}')
    exit(1)
"
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ SUCCESS: MobileCLIP is available through OpenCLIP!"
            echo ""
            echo "📝 To use MobileCLIP in your code:"
            echo "   import open_clip"
            echo "   model, _, preprocess = open_clip.create_model_and_transforms('MobileCLIP-S2', pretrained='datacompdr')"
            echo "   tokenizer = open_clip.get_tokenizer('MobileCLIP-S2')"
            echo ""
            echo "🎯 Available MobileCLIP models in OpenCLIP:"
            echo "   - MobileCLIP-S1"
            echo "   - MobileCLIP-S2" 
            echo "   - MobileCLIP-B"
            echo ""
            exit 0
        else
            echo "❌ All installation methods failed."
            exit 1
        fi
    }
else
    echo "✅ Found 'mobileclip' directory in repository"
fi

echo "📦 Installing requirements from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "⚠️  No requirements.txt found, proceeding anyway..."
fi

echo "📦 Installing MobileCLIP in editable mode..."
pip install -e .

# Go back to parent directory
cd ..

echo "🧪 Testing MobileCLIP installation..."
python -c "
import sys
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Test 1: Basic imports
try:
    import torch
    print('✅ PyTorch imported')
    
    import timm
    print('✅ timm imported')
    
    import transformers
    print('✅ transformers imported')
    
    import huggingface_hub
    print('✅ huggingface_hub imported')
    
except ImportError as e:
    print(f'❌ Basic import failed: {e}')
    sys.exit(1)

# Test 2: MobileCLIP import
try:
    import mobileclip
    print('✅ MobileCLIP imported successfully')
except ImportError as e:
    print(f'❌ MobileCLIP import failed: {e}')
    print('Available packages:')
    import pkg_resources
    installed_packages = [d.project_name for d in pkg_resources.working_set]
    mobileclip_packages = [p for p in installed_packages if 'clip' in p.lower()]
    print(f'CLIP-related packages: {mobileclip_packages}')
    sys.exit(1)

# Test 3: Model creation
try:
    model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained=False)
    print('✅ MobileCLIP model creation successful')
except Exception as e:
    print(f'❌ Model creation failed: {e}')
    sys.exit(1)

# Test 4: Tokenizer
try:
    tokenizer = mobileclip.get_tokenizer('mobileclip_s0')
    print('✅ MobileCLIP tokenizer creation successful')
except Exception as e:
    print(f'❌ Tokenizer creation failed: {e}')
    sys.exit(1)

# Test 5: Basic inference
try:
    test_image = torch.randn(1, 3, 224, 224)
    test_text = tokenizer(['test'])
    
    with torch.no_grad():
        image_features = model.encode_image(test_image)
        text_features = model.encode_text(test_text)
        
    print(f'✅ Model inference successful')
    print(f'   Image features shape: {image_features.shape}')
    print(f'   Text features shape: {text_features.shape}')
except Exception as e:
    print(f'❌ Model inference failed: {e}')
    sys.exit(1)

# Test 6: Reparameterization
try:
    from mobileclip.modules.common.mobileone import reparameterize_model
    mobile_model = reparameterize_model(model)
    print('✅ Model reparameterization successful')
except Exception as e:
    print(f'⚠️  Reparameterization failed: {e}')
    print('   This is optional - main functionality still works')

print('')
print('🎉 ALL TESTS PASSED! MobileCLIP is fully functional.')
print('')
print('Available models:')
try:
    models = ['mobileclip_s0', 'mobileclip_s1', 'mobileclip_s2', 'mobileclip_b', 'mobileclip_blt']
    for model_name in models:
        try:
            m, _, _ = mobileclip.create_model_and_transforms(model_name, pretrained=False)
            print(f'  ✅ {model_name}')
        except:
            print(f'  ❌ {model_name}')
except:
    pass
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎯 SUCCESS! MobileCLIP is fully installed and working."
    echo ""
    echo "📍 MobileCLIP source code location: $(pwd)/ml-mobileclip"
    echo ""
    echo "🚀 You can now use MobileCLIP:"
    echo ""
    echo "   import mobileclip"
    echo "   model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s0')"
    echo ""
    echo "🎯 Available models:"
    echo "   - mobileclip_s0 (fastest)"
    echo "   - mobileclip_s1"  
    echo "   - mobileclip_s2"
    echo "   - mobileclip_b"
    echo "   - mobileclip_blt (best accuracy)"
    echo ""
    echo "📚 Next steps:"
    echo "   1. Download models: python scripts/download/download_models.py --models mobileclip_s0"
    echo "   2. Convert models: python scripts/convert/convert_models.py --model mobileclip_s0"
    echo ""
else
    echo ""
    echo "❌ Installation verification failed."
    echo ""
    echo "🔧 Manual debugging steps:"
    echo ""
    echo "1. Check what got installed:"
    echo "   pip list | grep -i clip"
    echo "   pip show mobileclip"
    echo ""
    echo "2. Check repository structure:"
    echo "   cd $(pwd)/ml-mobileclip"
    echo "   ls -la"
    echo "   ls -la mobileclip/ 2>/dev/null || echo 'No mobileclip directory'"
    echo ""
    echo "3. Try manual import test:"
    echo "   cd $(pwd)/ml-mobileclip"
    echo "   python -c 'import sys; sys.path.insert(0, \".\"); import mobileclip'"
    echo ""
    echo "4. Check setup.py:"
    echo "   cat setup.py | grep -A5 -B5 find_packages"
    echo ""
    echo "🔄 Alternative: Try the OpenCLIP approach if direct installation fails."
    echo ""
    exit 1
fi

echo ""
echo "💡 Notes:"
echo "• MobileCLIP source is kept at: $(pwd)/ml-mobileclip"
echo "• This is an editable installation - changes to source affect the package"
echo "• For updates: cd $(pwd)/ml-mobileclip && git pull"
echo ""
echo "Happy coding! 🚀"