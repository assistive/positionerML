# DINOv2 Mobile Deployment

Complete deployment pipeline for DINOv2 vision transformer models on iOS and Android platforms.

## 🌟 Features

- ✅ Multi-platform support (iOS CoreML, Android TensorFlow Lite)
- ✅ Model optimization and quantization
- ✅ GPU acceleration (Neural Engine, NNAPI, GPU delegates)
- ✅ Ready-to-use mobile integration code
- ✅ Performance benchmarking and validation
- ✅ Complete deployment packages

## 📱 Supported Models

| Model | Parameters | Mobile Size | iOS Performance | Android Performance |
|-------|------------|-------------|-----------------|-------------------|
| DINOv2-ViT-S/14 | 22M | ~90MB | 100-200ms | 200-400ms |
| DINOv2-ViT-B/14 | 87M | ~350MB | 200-400ms | 400-800ms |
| DINOv2-ViT-L/14 | 304M | ~1.2GB | 500-1000ms | 1000-2000ms |

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda create -n dinov2_mobile python=3.9
conda activate dinov2_mobile

# Install dependencies
pip install torch torchvision
pip install coremltools  # For iOS
pip install tensorflow   # For Android
pip install pyyaml pillow numpy
```

### 2. Convert Models

```bash
# Convert DINOv2-ViT-S for both platforms
python scripts/convert/convert_dinov2.py --model dinov2_vits14 --platforms ios android

# Convert specific model for iOS only
python scripts/convert/convert_dinov2.py --model dinov2_vitb14 --platforms ios
```

### 3. Create Deployment Packages

```bash
# Create ready-to-use deployment packages
python scripts/deploy/deploy_mobile.py --platforms ios android --zip

# Packages will be created in ./deployment_packages/
```

### 4. Mobile Integration

#### iOS Integration
```swift
let inference = DINOv2Inference.shared

inference.extractFeatures(from: image) { result in
    switch result {
    case .success(let features):
        print("Features: \(features.count)")
    case .failure(let error):
        print("Error: \(error)")
    }
}
```

#### Android Integration
```kotlin
val inference = DINOv2Inference(context)
inference.initialize()

val features = inference.extractFeatures(bitmap)
features?.let {
    println("Features: ${it.size}")
}
```

## 📊 Performance Optimization

### iOS Optimization
- ✅ Neural Engine acceleration
- ✅ Float16 quantization
- ✅ Metal GPU acceleration
- ✅ Memory optimization

### Android Optimization
- ✅ NNAPI acceleration
- ✅ GPU delegate support
- ✅ INT8 quantization
- ✅ XNNPACK optimization

## 🛠️ Development

### Project Structure
```
dinov2_mobile/
├── config/                 # Configuration files
├── src/                    # Source code
│   ├── core/              # Core DINOv2 implementation
│   ├── mobile_converter/  # Mobile conversion tools
│   └── utils/             # Utilities
├── scripts/               # Automation scripts
├── mobile/                # Platform-specific code
│   ├── ios/              # iOS Swift integration
│   └── android/          # Android Kotlin integration
├── models/               # Model storage
├── examples/             # Example applications
└── docs/                 # Documentation
```

### Custom Configuration

Edit `config/dinov2_config.yaml` to customize:
- Model variants and parameters
- Quantization settings
- Platform-specific optimizations
- Performance targets

## 📖 Documentation

- [iOS Deployment Guide](docs/ios_deployment.md)
- [Android Deployment Guide](docs/android_deployment.md)
- [Performance Optimization](docs/performance_optimization.md)
- [Troubleshooting](docs/troubleshooting.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Meta Research for DINOv2
- Apple for CoreML framework
- Google for TensorFlow Lite
- Open source community

---

**Ready to deploy DINOv2 on mobile? Get started with the quick start guide above! 🚀**
