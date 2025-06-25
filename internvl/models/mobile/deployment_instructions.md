# InternVL Mobile Deployment Instructions

Generated on: 2025-06-25T10:53:39

## Converted Models


### IOS Deployment

**Model Path:** `models/mobile/ios/internvl_mobile.mlpackage`

**Integration Steps:**

1. **Add Model to Xcode Project:**
   ```
   Drag the .mlmodel file into your Xcode project
   ```

2. **Swift Code Example:**
   ```swift
   import CoreML
   import Vision
   
   // Load the model
   guard let model = try? InternVLMobile(configuration: MLModelConfiguration()) else {
       fatalError("Failed to load model")
   }
   
   // Prepare input
   let pixelBuffer = // Your image as CVPixelBuffer
   let textTokens = // Your tokenized text as MLMultiArray
   
   // Make prediction
   let prediction = try model.prediction(image: pixelBuffer, input_ids: textTokens)
   ```

3. **Required Frameworks:**
   - CoreML
   - Vision
   - Accelerate (for preprocessing)

### ANDROID Deployment

**Model Path:** `models/mobile/android/internvl_mobile.tflite`

**Integration Steps:**

1. **Add to Android Project:**
   ```
   Place the .tflite file in app/src/main/assets/
   ```

2. **Add Dependencies (app/build.gradle):**
   ```gradle
   implementation 'org.tensorflow:tensorflow-lite:2.13.0'
   implementation 'org.tensorflow:tensorflow-lite-gpu:2.13.0'
   implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
   ```

3. **Kotlin/Java Code Example:**
   ```kotlin
   import org.tensorflow.lite.Interpreter
   import org.tensorflow.lite.support.image.TensorImage
   
   // Load the model
   val tfliteModel = loadModelFile("model.tflite")
   val interpreter = Interpreter(tfliteModel)
   
   // Prepare inputs
   val imageInput = TensorImage.fromBitmap(bitmap)
   val textInput = // Your tokenized text as IntArray
   
   // Run inference
   val outputs = arrayOf(FloatArray(outputSize))
   interpreter.run(arrayOf(imageInput.buffer, textInput), outputs)
   ```

## Performance Optimization

### iOS:
- Use Neural Engine when available
- Enable compute precision optimization
- Consider model quantization for smaller size

### Android:
- Use GPU delegate for faster inference
- Enable NNAPI delegate when supported
- Apply dynamic quantization

## Integration Notes

1. **Preprocessing**: Ensure input images are properly normalized and resized
2. **Tokenization**: Use the same tokenizer as during training
3. **Postprocessing**: Apply appropriate output formatting for your use case
4. **Error Handling**: Implement proper error handling for model loading and inference

## Troubleshooting

### Common Issues:

1. **Model loading fails**:
   - Check file path and permissions
   - Verify model format compatibility

2. **Inference errors**:
   - Validate input shapes and types
   - Ensure proper preprocessing

3. **Performance issues**:
   - Enable hardware acceleration
   - Consider model optimization
