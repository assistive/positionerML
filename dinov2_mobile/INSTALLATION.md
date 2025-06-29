# DINOv2 Mobile Deployment - Installation Guide

## 🚀 Quick Setup

### Option 1: Automated Setup (Recommended)
```bash
# 1. Clone/create the directory structure
./dinov2_mobile_structure.sh

# 2. Navigate to the directory
cd dinov2_mobile

# 3. Run the automated setup
python setup.py
```

### Option 2: Manual Installation

#### Step 1: Create Environment
```bash
conda create -n dinov2_mobile python=3.9
conda activate dinov2_mobile
```

#### Step 2: Install Core Dependencies
```bash
# PyTorch (choose based on your system)
pip install torch torchvision

# For CUDA support (optional):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Step 3: Install Mobile Conversion Tools
```bash
# iOS conversion (CoreML)
pip install coremltools

# Android conversion (TensorFlow)
pip install tensorflow

# For macOS with Apple Silicon:
# pip install tensorflow-macos tensorflow-metal
```

#### Step 4: Install Additional Dependencies
```bash
# Model conversion utilities
pip install onnx onnxruntime onnx-tf

# General utilities
pip install pyyaml pillow numpy pathlib
```

## 🔍 Verification

### Check Installation
```bash
python -c "
import torch; print(f'✅ PyTorch: {torch.__version__}')
import coremltools; print(f'✅ CoreML: {coremltools.__version__}')  
import tensorflow; print(f'✅ TensorFlow: {tensorflow.__version__}')
import onnx; print(f'✅ ONNX: {onnx.__version__}')
print('🎉 All dependencies installed successfully!')
"
```

### Test Model Loading
```bash
python -c "
import torch
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
print('✅ DINOv2 model loaded successfully')
"
```

## ⚡ Quick Start Commands

After installation, test the conversion:

```bash
# Convert DINOv2-ViT-S for iOS (safest option)
python scripts/convert/convert_dinov2_enhanced.py \
  --model dinov2_vits14 \
  --platforms ios \
  --cpu-only

# Convert for both platforms
python scripts/convert/convert_dinov2_enhanced.py \
  --model dinov2_vits14 \
  --platforms ios android \
  --auto-install

# Create deployment packages
python scripts/deploy/deploy_mobile.py --zip
```

## 🆘 Troubleshooting

If you encounter issues during installation:

1. **Check Python version:**
   ```bash
   python --version  # Should be 3.9+
   ```

2. **Update pip:**
   ```bash
   pip install --upgrade pip
   ```

3. **Clear pip cache:**
   ```bash
   pip cache purge
   ```

4. **Use conda for problematic packages:**
   ```bash
   conda install pytorch torchvision -c pytorch
   ```

5. **Check the troubleshooting guide:**
   ```bash
   cat TROUBLESHOOTING.md
   ```

Ready to convert DINOv2 for mobile deployment! 🚀
