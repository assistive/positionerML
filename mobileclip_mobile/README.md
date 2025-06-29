# MobileCLIP Mobile Deployment

Complete deployment pipeline for MobileCLIP image-text models on iOS and Android platforms.

## 🌟 Features

- ✅ Multi-platform support (iOS CoreML, Android TensorFlow Lite)
- ✅ All MobileCLIP variants (S0, S1, S2, B, B-LT)
- ✅ Model optimization and quantization
- ✅ GPU acceleration (Neural Engine, NNAPI, GPU delegates)
- ✅ Ready-to-use mobile integration code (Swift/Kotlin)
- ✅ Zero-shot image classification capabilities
- ✅ Performance benchmarking and validation

## 📱 Supported Models

| Model | Parameters | Mobile Size | iOS Performance | Android Performance | Accuracy |
|-------|------------|-------------|-----------------|-------------------|----------|
| MobileCLIP-S0 | 53.8M | ~55MB | 30-50ms | 80-120ms | 67.8% ImageNet |
| MobileCLIP-S1 | 67.6M | ~80MB | 50-80ms | 120-180ms | 72.6% ImageNet |
| MobileCLIP-S2 | 78.1M | ~95MB | 70-100ms | 150-220ms | 74.4% ImageNet |
| MobileCLIP-B | 128.7M | ~150MB | 120-200ms | 300-450ms | 76.8% ImageNet |
| MobileCLIP-B(LT) | 128.7M | ~150MB | 120-200ms | 300-450ms | 77.2% ImageNet |

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda create -n mobileclip_mobile python=3.10
conda activate mobileclip_mobile

# Install dependencies
pip install torch torchvision
pip install git+https://github.com/apple/ml-mobileclip
pip install coremltools  # For iOS
pip install tensorflow   # For Android
pip install pyyaml pillow numpy huggingface_hub
```

### 2. Download Models

```bash
# Download MobileCLIP-S0 (fastest, smallest)
python scripts/download/download_models.py --models mobileclip_s0

# Download multiple models
python scripts/download/download_models.py --models mobileclip_s0 mobileclip_s1 mobileclip_s2

# Download from Hugging Face
python scripts/download/download_models.py --source huggingface --models mobileclip_s1
```

### 3. Convert Models

```bash
# Convert MobileCLIP-S0 for both platforms
python scripts/convert/convert_models.py --model mobileclip_s0 --platforms ios android

# Convert specific model for iOS only
python scripts/convert/convert_models.py --model mobileclip_s1 --platforms ios
```

### 4. Create Deployment Packages

```bash
# Create ready-to-use deployment packages
python scripts/deploy/deploy_mobile.py --platforms ios android --zip

# Packages will be created in ./deployment/packages/
```

## 📱 Mobile Integration

### iOS Integration (Swift)

```swift
import UIKit

class ViewController: UIViewController {
    private let mobileCLIP = MobileCLIPInference.shared
    
    func classifyImage(_ image: UIImage) {
        let labels = ["dog", "cat", "bird", "car", "plane"]
        
        mobileCLIP.zeroShotClassify(image: image, labels: labels) { result in
            switch result {
            case .success(let results):
                for (label, confidence) in results.prefix(3) {
                    print("\(label): \(confidence)")
                }
            case .failure(let error):
                print("Error: \(error)")
            }
        }
    }
}
```

### Android Integration (Kotlin)

```kotlin
class MainActivity : AppCompatActivity() {
    private lateinit var mobileCLIP: MobileCLIPInference
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        mobileCLIP = MobileCLIPInference(this)
        
        lifecycleScope.launch {
            if (mobileCLIP.initialize()) {
                classifyImage(bitmap)
            }
        }
    }
    
    private suspend fun classifyImage(bitmap: Bitmap) {
        val labels = listOf("dog", "cat", "bird", "car", "plane")
        val results = mobileCLIP.zeroShotClassify(bitmap, labels)
        
        results?.take(3)?.forEach { (label, confidence) ->
            Log.d("Classification", "$label: $confidence")
        }
    }
}
```

## 🎯 Performance Optimization

### iOS Optimization
- ✅ Neural Engine acceleration
- ✅ Float16 quantization
- ✅ Metal GPU acceleration
- ✅ Memory optimization
- ✅ CoreML optimization

### Android Optimization
- ✅ NNAPI acceleration
- ✅ GPU delegate support
- ✅ INT8 quantization
- ✅ XNNPACK optimization
- ✅ Multi-threading support

## 🛠️ Development

### Project Structure
```
mobileclip_mobile/
├── config/                 # Configuration files
│   ├── mobileclip_config.yaml
│   └── mobile/            # Platform-specific configs
├── src/                   # Source code
│   ├── core/             # Core MobileCLIP implementation
│   ├── mobile_converter/ # Mobile conversion tools
│   └── utils/            # Utilities
├── scripts/              # Automation scripts
│   ├── download/         # Model download scripts
│   ├── convert/          # Conversion scripts
│   └── deploy/           # Deployment scripts
├── mobile/               # Platform-specific code
│   ├── ios/             # iOS Swift integration
│   └── android/         # Android Kotlin integration
├── models/              # Model storage
│   ├── pretrained/      # Downloaded models
│   └── converted/       # Converted models
├── examples/            # Example applications
└── docs/                # Documentation
```

### Custom Configuration

Edit `config/mobileclip_config.yaml` to customize:
- Model variants and parameters
- Quantization settings
- Platform-specific optimizations
- Performance targets

## 📊 Benchmarking

### Performance Testing

```bash
# Run performance benchmarks
python tools/benchmarking/mobile_benchmark.py --model mobileclip_s0 --platform ios
python tools/benchmarking/mobile_benchmark.py --model mobileclip_s1 --platform android

# Memory usage analysis
python tools/benchmarking/memory_benchmark.py --model mobileclip_s0
```

### Accuracy Validation

```bash
# Validate conversion accuracy
python tools/benchmarking/accuracy_benchmark.py --original-model mobileclip_s0.pt --converted-model mobileclip_s0.tflite
```

## 🔧 Advanced Usage

### Custom Model Training

```python
# Fine-tune MobileCLIP on custom data
from src.core.mobileclip_model import MobileCLIPModel

model = MobileCLIPModel("mobileclip_s0")
# Add your fine-tuning code here
```

### Batch Processing

```python
# Process multiple images efficiently
images = [image1, image2, image3]
labels = ["dog", "cat", "bird"]

for image in images:
    results = model.zero_shot_classify(image, labels)
    print(results)
```

## 📖 Documentation

- [iOS Deployment Guide](docs/ios_deployment.md)
- [Android Deployment Guide](docs/android_deployment.md)
- [Performance Optimization](docs/performance_optimization.md)
- [Troubleshooting](docs/troubleshooting.md)
- [API Reference](docs/api_reference.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Apple Research for MobileCLIP
- Apple for CoreML framework
- Google for TensorFlow Lite
- Hugging Face for model hosting
- Open source community

## 🆘 Troubleshooting

### Common Issues

1. **Model download fails:**
   ```bash
   # Check internet connection and try again
   python scripts/download/download_models.py --force
   ```

2. **CoreML conversion fails:**
   ```bash
   # Ensure you're on macOS with latest Xcode
   pip install --upgrade coremltools
   ```

3. **TensorFlow Lite conversion fails:**
   ```bash
   # Install required dependencies
   pip install tensorflow onnx onnx-tf
   ```

4. **iOS app crashes:**
   - Ensure model files are added to Xcode project
   - Check iOS deployment target (15.0+)
   - Verify model file paths

5. **Android inference slow:**
   - Enable NNAPI acceleration
   - Use GPU delegate if available
   - Check if model is quantized

### Getting Help

- Check the [documentation](docs/)
- Open an issue on GitHub
- Join our community discussions

---

**Ready to deploy MobileCLIP on mobile? Get started with the quick start guide above! 🚀**
