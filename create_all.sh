#!/bin/bash

# Script placement and integration guide for the Unified VLM Mobile Converter

echo "Setting up Unified VLM Mobile Converter in your project structure..."

# Option 1: As a standalone tool in the root directory (RECOMMENDED)
echo "📁 Option 1: Root-level standalone tool (RECOMMENDED)"
echo "Place the script at the root of your VLM projects directory:"
echo ""
echo "your-vlm-workspace/"
echo "├── unified_mobile_converter.py     ← PLACE THE SCRIPT HERE"
echo "├── requirements_mobile.txt         ← Dependencies for mobile conversion"
echo "├── mobile_config.yaml             ← Configuration file"
echo "├── fastvlm/                       ← Your existing FastVLM project"
echo "│   ├── models/pretrained/"
echo "│   └── src/"
echo "├── internvl/                      ← Your existing InternVL project"
echo "│   ├── models/pretrained/"
echo "│   └── src/"
echo "├── qwen-vl-service/              ← Your Qwen 2.5-VL project"
echo "│   ├── models/pretrained/"
echo "│   └── src/"
echo "├── mobile_models/                ← Generated mobile models (auto-created)"
echo "│   ├── ios/"
echo "│   └── android/"
echo "└── conversion_reports/           ← Conversion reports and logs"
echo ""

# Create the recommended structure
mkdir -p mobile_models/{ios,android}
mkdir -p conversion_reports

# Option 2: Inside each project as a shared tool
echo "📁 Option 2: Shared tool within projects"
echo "Place copies in each VLM project's tools directory:"
echo ""
echo "fastvlm/"
echo "├── tools/"
echo "│   └── unified_mobile_converter.py"
echo "internvl/"
echo "├── tools/"
echo "│   └── unified_mobile_converter.py"
echo "qwen-vl-service/"
echo "├── tools/"
echo "│   └── unified_mobile_converter.py"
echo ""

# Option 3: As a separate package
echo "📁 Option 3: Separate package/repository"
echo "Create a dedicated mobile conversion package:"
echo ""
echo "vlm-mobile-converter/"
echo "├── unified_mobile_converter.py"
echo "├── setup.py"
echo "├── requirements.txt"
echo "└── README.md"
echo ""

# Create the main requirements file for mobile conversion
cat > requirements_mobile.txt << 'EOF'
# Core dependencies for Unified VLM Mobile Converter
torch>=2.0.0
transformers>=4.49.0
accelerate>=0.20.0

# Model-specific dependencies
qwen-vl-utils[decord]==0.0.8

# Mobile conversion frameworks
onnx>=1.14.0
onnxruntime>=1.15.0

# iOS conversion (macOS only)
coremltools>=7.0; sys_platform == "darwin"

# Android conversion
tensorflow>=2.13.0

# Data processing
pillow>=9.0.0
opencv-python>=4.5.0
numpy>=1.24.0
pyyaml>=6.0

# Utilities
huggingface-hub>=0.16.0
psutil>=5.9.0
tqdm>=4.65.0
EOF

# Create configuration file
cat > mobile_config.yaml << 'EOF'
# Unified VLM Mobile Converter Configuration

# Global settings
output_directory: "mobile_models"
cache_directory: "model_cache"
log_level: "INFO"
max_parallel_workers: 2

# Model discovery paths
model_search_paths:
  - "fastvlm/models/pretrained"
  - "internvl/models/pretrained" 
  - "qwen-vl-service/models/pretrained"
  - "models/pretrained"
  - "~/.cache/huggingface/hub"

# Conversion settings
conversion:
  platforms:
    - "ios"
    - "android"
  
  optimization:
    quantization_bits: 8
    enable_pruning: true
    pruning_sparsity: 0.3
    
  mobile_specific:
    ios:
      compute_units: "cpuAndNeuralEngine"
      minimum_deployment_target: "iOS15"
      enable_metal_performance_shaders: true
      
    android:
      optimization_level: "default"
      enable_nnapi: true
      target_api_level: 24

# Model-specific overrides
model_overrides:
  qwen-2.5-vl-3b:
    quantization_bits: 8
    enable_pruning: true
    
  qwen-2.5-vl-7b:
    quantization_bits: 8
    enable_pruning: false  # Preserve accuracy
    
  fastvlm-tiny:
    quantization_bits: 4   # Aggressive optimization
    enable_pruning: true
    
  internvl2-2b:
    quantization_bits: 8
    enable_pruning: true

# Hardware requirements check
hardware_requirements:
  minimum_ram_gb: 16
  recommended_ram_gb: 32
  gpu_memory_gb: 8
  disk_space_gb: 100
EOF

# Create setup script for the unified converter
cat > setup_mobile_converter.py << 'EOF'
#!/usr/bin/env python3
"""
Setup script for the Unified VLM Mobile Converter
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_system_requirements():
    """Check if system meets requirements."""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {python_version.major}.{python_version.minor}")
    
    # Check available memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 16:
            print(f"⚠️  Low memory: {memory_gb:.1f}GB (16GB+ recommended)")
        else:
            print(f"✅ Memory: {memory_gb:.1f}GB")
    except ImportError:
        print("⚠️  Cannot check memory (psutil not installed)")
    
    # Check disk space
    disk_free = os.statvfs('.').f_bavail * os.statvfs('.').f_frsize / (1024**3)
    if disk_free < 100:
        print(f"⚠️  Low disk space: {disk_free:.1f}GB (100GB+ recommended)")
    else:
        print(f"✅ Disk space: {disk_free:.1f}GB available")
    
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    # Core dependencies
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-r", "requirements_mobile.txt"
    ], check=True)
    
    # Platform-specific dependencies
    if platform.system() == "Darwin":  # macOS
        print("🍎 Installing iOS development tools...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "coremltools"
        ])
    
    print("✅ Dependencies installed successfully")

def setup_directories():
    """Create necessary directories."""
    print("\n📁 Setting up directories...")
    
    directories = [
        "mobile_models",
        "mobile_models/ios", 
        "mobile_models/android",
        "model_cache",
        "conversion_reports",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  Created: {directory}")

def main():
    """Main setup function."""
    print("🚀 Setting up Unified VLM Mobile Converter\n")
    
    try:
        if not check_system_requirements():
            print("❌ System requirements not met")
            return 1
        
        install_dependencies()
        setup_directories()
        
        print("\n✅ Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Place your VLM models in the respective directories:")
        print("   - fastvlm/models/pretrained/")
        print("   - internvl/models/pretrained/")
        print("   - qwen-vl-service/models/pretrained/")
        print("\n2. Run the converter:")
        print("   python unified_mobile_converter.py --discover-only")
        print("   python unified_mobile_converter.py --all")
        print("\n3. Check results in mobile_models/ directory")
        
        return 0
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

# Create a wrapper script for easy execution
cat > convert_all_vlms.sh << 'EOF'
#!/bin/bash

# Unified VLM Mobile Converter - Easy execution script

echo "🚀 Unified VLM Mobile Converter"
echo "================================"

# Check if the main script exists
if [ ! -f "unified_mobile_converter.py" ]; then
    echo "❌ unified_mobile_converter.py not found!"
    echo "Please place the script in the current directory."
    exit 1
fi

# Set default options
PLATFORMS="ios android"
QUANTIZATION_BITS=8
ENABLE_PRUNING="--enable-pruning"
OUTPUT_DIR="mobile_models"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --ios-only)
            PLATFORMS="ios"
            shift
            ;;
        --android-only)
            PLATFORMS="android"
            shift
            ;;
        --4bit)
            QUANTIZATION_BITS=4
            shift
            ;;
        --16bit)
            QUANTIZATION_BITS=16
            shift
            ;;
        --no-pruning)
            ENABLE_PRUNING="--disable-pruning"
            shift
            ;;
        --discover)
            python unified_mobile_converter.py --discover-only
            exit 0
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --ios-only      Convert only for iOS"
            echo "  --android-only  Convert only for Android"
            echo "  --4bit         Use 4-bit quantization"
            echo "  --16bit        Use 16-bit quantization"
            echo "  --no-pruning   Disable model pruning"
            echo "  --discover     Only discover available models"
            echo "  --help         Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for available options"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the conversion
echo "Starting conversion with options:"
echo "  Platforms: $PLATFORMS"
echo "  Quantization: ${QUANTIZATION_BITS}-bit"
echo "  Pruning: $(echo $ENABLE_PRUNING | grep -q disable && echo 'disabled' || echo 'enabled')"
echo "  Output: $OUTPUT_DIR"
echo ""

python unified_mobile_converter.py \
    --all \
    --platforms $PLATFORMS \
    --quantization-bits $QUANTIZATION_BITS \
    $ENABLE_PRUNING \
    --output-dir "$OUTPUT_DIR" \
    --verbose

# Check results
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Conversion completed successfully!"
    echo "📁 Results saved in: $OUTPUT_DIR"
    echo "📊 Check conversion_report.json for detailed metrics"
    
    # Show quick summary
    echo ""
    echo "📱 Mobile models generated:"
    find "$OUTPUT_DIR" -name "*.onnx" -o -name "*.mlpackage" -o -name "*.tflite" | sort
else
    echo ""
    echo "❌ Conversion failed!"
    echo "📋 Check the logs for error details"
fi
EOF

# Make scripts executable
chmod +x convert_all_vlms.sh
chmod +x setup_mobile_converter.py

# Create integration guide
cat > MOBILE_CONVERTER_INTEGRATION.md << 'EOF'
# Unified VLM Mobile Converter - Integration Guide

## 📍 Where to Place the Script

### Option 1: Root-Level Standalone (RECOMMENDED)

Place `unified_mobile_converter.py` at the root of your VLM workspace:

```
your-vlm-workspace/
├── unified_mobile_converter.py     ← Main script here
├── requirements_mobile.txt         ← Mobile dependencies
├── mobile_config.yaml             ← Configuration
├── setup_mobile_converter.py      ← Setup script
├── convert_all_vlms.sh            ← Easy wrapper script
├── fastvlm/                       ← Existing projects
├── internvl/
├── qwen-vl-service/
└── mobile_models/                 ← Output directory
```

**Advantages:**
- Single script manages all VLM projects
- Shared configuration and output
- Easy maintenance and updates
- Cross-project model discovery

### Option 2: Within Each Project

Place copies in each project's tools directory:

```
fastvlm/tools/unified_mobile_converter.py
internvl/tools/unified_mobile_converter.py
qwen-vl-service/tools/unified_mobile_converter.py
```

**Advantages:**
- Project-specific customization
- Independent execution
- No cross-dependencies

### Option 3: Separate Package

Create a dedicated repository:

```
vlm-mobile-converter/
├── unified_mobile_converter.py
├── setup.py
├── requirements.txt
└── README.md
```

## 🚀 Quick Setup

### 1. Initial Setup

```bash
# Place the script files in your workspace root
cd your-vlm-workspace/

# Run setup (installs dependencies, creates directories)
python setup_mobile_converter.py

# Verify installation
python unified_mobile_converter.py --discover-only
```

### 2. Easy Conversion

```bash
# Convert all models (default: iOS + Android, 8-bit, with pruning)
./convert_all_vlms.sh

# iOS only with 4-bit quantization
./convert_all_vlms.sh --ios-only --4bit

# Android only without pruning
./convert_all_vlms.sh --android-only --no-pruning

# High-quality conversion (16-bit, no pruning)
./convert_all_vlms.sh --16bit --no-pruning
```

### 3. Manual Control

```bash
# Discover available models
python unified_mobile_converter.py --discover-only

# Convert specific models
python unified_mobile_converter.py --models qwen-2.5-vl-3b fastvlm-tiny

# Custom configuration
python unified_mobile_converter.py \
    --all \
    --platforms ios android \
    --quantization-bits 8 \
    --enable-pruning \
    --max-workers 4 \
    --output-dir custom_mobile_models
```

## 📁 Directory Structure After Setup

```
your-vlm-workspace/
├── unified_mobile_converter.py     # Main converter script
├── requirements_mobile.txt         # Dependencies
├── mobile_config.yaml             # Configuration
├── setup_mobile_converter.py      # Setup utility
├── convert_all_vlms.sh            # Easy wrapper
├── 
├── fastvlm/                       # Your existing projects
│   └── models/pretrained/         # Models discovered here
├── internvl/
│   └── models/pretrained/
├── qwen-vl-service/
│   └── models/pretrained/
├── 
├── mobile_models/                 # 📱 GENERATED MOBILE MODELS
│   ├── ios/
│   │   ├── qwen-2.5-vl-3b.mlpackage/
│   │   ├── fastvlm-tiny.mlpackage/
│   │   └── internvl2-2b.mlpackage/
│   ├── android/
│   │   ├── qwen-2.5-vl-3b.tflite
│   │   ├── fastvlm-tiny.tflite
│   │   └── internvl2-2b.tflite
│   └── cross-platform/
│       ├── qwen-2.5-vl-3b.onnx
│       ├── fastvlm-tiny.onnx
│       └── internvl2-2b.onnx
├── 
├── conversion_reports/            # Reports and metrics
│   ├── conversion_report.json
│   └── mobile_conversion.log
└── model_cache/                  # Temporary files
```

## 🔧 Configuration

Edit `mobile_config.yaml` to customize:

```yaml
# Model discovery paths
model_search_paths:
  - "your-custom-path/models"
  - "another-project/pretrained"

# Default conversion settings
conversion:
  quantization_bits: 8
  enable_pruning: true
  platforms: ["ios", "android"]

# Model-specific overrides
model_overrides:
  your-custom-model:
    quantization_bits: 4
    enable_pruning: false
```

## 📱 Using Converted Models

### iOS Integration

```swift
import CoreML

// Load converted model
let modelURL = Bundle.main.url(forResource: "qwen-2.5-vl-3b", withExtension: "mlpackage")!
let model = try MLModel(contentsOf: modelURL)

// Use in your iOS app
let coreMLModel = try VNCoreMLModel(for: model)
```

### Android Integration

```kotlin
import org.tensorflow.lite.Interpreter

// Load converted model
val model = loadModelFile(context, "qwen-2.5-vl-3b.tflite")
val interpreter = Interpreter(model)

// Use in your Android app
interpreter.run(inputData, outputData)
```

## 🚨 Troubleshooting

### Common Issues

1. **Script not found**: Ensure `unified_mobile_converter.py` is in the correct location
2. **No models discovered**: Check that models are in the search paths
3. **Conversion failures**: Install platform-specific dependencies:
   ```bash
   # For iOS (macOS only)
   pip install coremltools
   
   # For Android
   pip install tensorflow
   ```

### Getting Help

```bash
# Check available options
python unified_mobile_converter.py --help

# Discover models
python unified_mobile_converter.py --discover-only

# Verbose output for debugging
python unified_mobile_converter.py --all --verbose
```

## 🎯 Recommended Workflow

1. **Setup once**: Run `python setup_mobile_converter.py`
2. **Train/download models**: Place in respective project directories
3. **Convert regularly**: Use `./convert_all_vlms.sh` after model updates
4. **Deploy**: Use generated mobile models in your apps
5. **Monitor**: Check conversion reports for optimization metrics

This setup gives you a **single command to convert all your VLMs to mobile formats** while maintaining clean project organization! 🚀
EOF

echo ""
echo "✅ Unified Mobile Converter placement guide created!"
echo ""
echo "📍 RECOMMENDED PLACEMENT:"
echo "Place 'unified_mobile_converter.py' at the ROOT of your VLM workspace"
echo ""
echo "📁 Your structure should look like:"
echo "your-vlm-workspace/"
echo "├── unified_mobile_converter.py     ← MAIN SCRIPT HERE"
echo "├── requirements_mobile.txt         ← Dependencies"
echo "├── mobile_config.yaml             ← Configuration"
echo "├── setup_mobile_converter.py      ← Setup utility"
echo "├── convert_all_vlms.sh            ← Easy wrapper"
echo "├── fastvlm/                       ← Your existing projects"
echo "├── internvl/"
echo "├── qwen-vl-service/"
echo "└── mobile_models/                 ← Generated mobile models"
echo ""
echo "🚀 Quick start:"
echo "1. python setup_mobile_converter.py"
echo "2. ./convert_all_vlms.sh"
echo ""
echo "📖 See MOBILE_CONVERTER_INTEGRATION.md for detailed setup guide"
