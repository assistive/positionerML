# iOS Integration Guide

## Requirements
- iOS 15.0 or later
- Xcode 13.0 or later
- CoreML framework

## Setup

1. **Add Model to Project**:
   - Drag `internvl_mobile.mlpackage` into your Xcode project
   - Ensure it's added to your target
   - **Note**: This is an ML Program model (.mlpackage), not a Neural Network model (.mlmodel)

2. **Add Framework**:
   ```swift
   import CoreML
   import Vision
   ```

3. **Basic Usage**:
   ```swift
   let vlModel = InternVLMobile()
   
   // Prepare your inputs as MLMultiArray
   let pixelValues = // Your image tensor as MLMultiArray [1, 3, 224, 224]
   let inputIds = // Your text tokens as MLMultiArray [1, 512]
   let attentionMask = // Your attention mask as MLMultiArray [1, 512]
   
   // Make prediction
   if let result = vlModel.predict(
       pixelValues: pixelValues, 
       inputIds: inputIds, 
       attentionMask: attentionMask
   ) {
       // Handle the result [1, 512, 768]
       print("Prediction successful")
   }
   ```

## Input Processing

### Image Processing (to MLMultiArray):
```swift
func preprocessImage(_ image: UIImage) -> MLMultiArray? {
    // Resize to 224x224
    let targetSize = CGSize(width: 224, height: 224)
    UIGraphicsBeginImageContextWithOptions(targetSize, true, 1.0)
    image.draw(in: CGRect(origin: .zero, size: targetSize))
    let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
    UIGraphicsEndImageContext()
    
    // Convert to MLMultiArray [1, 3, 224, 224]
    guard let cgImage = resizedImage?.cgImage else { return nil }
    
    let shape = [1, 3, 224, 224] as [NSNumber]
    guard let mlArray = try? MLMultiArray(shape: shape, dataType: .float32) else {
        return nil
    }
    
    // Fill with RGB pixel values (normalized to [0, 1])
    // Implementation depends on your specific preprocessing needs
    
    return mlArray
}
```

### Text Processing (to MLMultiArray):
```swift
func preprocessText(_ text: String) -> (MLMultiArray?, MLMultiArray?) {
    // Implement your tokenization logic here
    let tokens = tokenize(text) // Your tokenizer implementation
    
    guard let inputIds = try? MLMultiArray(shape: [1, 512], dataType: .int32),
          let attentionMask = try? MLMultiArray(shape: [1, 512], dataType: .int32) else {
        return (nil, nil)
    }
    
    // Fill arrays with token data
    for (index, token) in tokens.enumerated() {
        if index < 512 {
            inputIds[index] = NSNumber(value: token)
            attentionMask[index] = NSNumber(value: 1)
        }
    }
    
    // Pad remaining positions
    for index in tokens.count..<512 {
        inputIds[index] = NSNumber(value: 0)  // PAD token
        attentionMask[index] = NSNumber(value: 0)
    }
    
    return (inputIds, attentionMask)
}
```

## Model Information

- **Format**: CoreML ML Program (.mlpackage)
- **Precision**: Float16 (optimized for mobile)
- **Input Shapes**:
  - `pixel_values`: [1, 3, 224, 224] (Float32)
  - `input_ids`: [1, 512] (Int32)
  - `attention_mask`: [1, 512] (Int32)
- **Output Shape**: [1, 512, 768] (Float16)

## Performance Tips

- Use background queues for model inference
- Cache the model instance
- Consider using Vision framework for image preprocessing
- Implement proper error handling
- The model uses Neural Engine when available for optimal performance

## Troubleshooting

- **Model loading fails**: Ensure the .mlpackage is correctly added to your Xcode target
- **Inference errors**: Verify input shapes match exactly [1, 3, 224, 224], [1, 512], [1, 512]
- **Performance issues**: Monitor memory usage and consider reducing batch processing
