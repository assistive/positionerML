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
