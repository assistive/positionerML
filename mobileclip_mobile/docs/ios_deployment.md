# iOS Deployment Guide

Complete guide for deploying MobileCLIP models on iOS devices.

## Prerequisites

- macOS development machine
- Xcode 15.0 or later
- iOS 15.0 or later target
- Device with Neural Engine (recommended)

## Model Conversion

Convert your MobileCLIP model to CoreML format:

```bash
python scripts/convert/convert_models.py --model mobileclip_s0 --platforms ios
```

This creates CoreML model packages in `models/converted/`:
- `mobileclip_s0_image.mlpackage` - Image encoder
- `mobileclip_s0_text.mlpackage` - Text encoder

## Xcode Integration

### 1. Add Models to Project

1. Drag both `.mlpackage` files into your Xcode project
2. Ensure "Add to target" is checked for your app target
3. Verify models appear in your project navigator

### 2. Add Swift Integration Class

1. Add `MobileCLIPInference.swift` to your project
2. Import required frameworks in your app:

```swift
import CoreML
import Vision
import Accelerate
```

### 3. Basic Usage

```swift
class ViewController: UIViewController {
    private let mobileCLIP = MobileCLIPInference.shared
    
    @IBAction func classifyImage(_ sender: UIButton) {
        guard let image = imageView.image else { return }
        
        let labels = ["dog", "cat", "bird", "car", "airplane"]
        
        mobileCLIP.zeroShotClassify(image: image, labels: labels) { result in
            DispatchQueue.main.async {
                switch result {
                case .success(let results):
                    self.displayResults(results)
                case .failure(let error):
                    self.showError(error)
                }
            }
        }
    }
    
    private func displayResults(_ results: [(label: String, confidence: Float)]) {
        for (label, confidence) in results.prefix(3) {
            print("\(label): \(String(format: "%.2f", confidence * 100))%")
        }
    }
}
```

## Performance Optimization

### Neural Engine Acceleration

Models automatically use Neural Engine when available:

```swift
// Verify Neural Engine usage
if let computeUnits = model.configuration.computeUnits {
    print("Compute units: \(computeUnits)")
}
```

### Memory Management

```swift
// Process images in background queue
DispatchQueue.global(qos: .userInitiated).async {
    self.mobileCLIP.extractImageFeatures(from: image) { result in
        DispatchQueue.main.async {
            // Update UI
        }
    }
}
```

## Troubleshooting

### Common Issues

1. **Model not found error:**
   - Verify `.mlpackage` files are added to project target
   - Check model file names match code expectations

2. **Slow inference:**
   - Ensure Neural Engine is available (iPhone 12+, iPad Air 4+)
   - Check model is using float16 precision

3. **Memory warnings:**
   - Process images on background queue
   - Resize large images before processing

### Performance Tips

- Use Neural Engine compatible devices
- Resize images to 224x224 before processing
- Batch process multiple images when possible
- Cache model instances to avoid reload overhead

## Example App

See `examples/ios/MobileCLIPDemo/` for a complete example application.
