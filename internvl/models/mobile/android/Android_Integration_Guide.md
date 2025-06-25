# Android Integration Guide

## Requirements
- Android API 21 (Android 5.0) or later
- TensorFlow Lite 2.13.0 or later

## Setup

1. **Add Dependencies** (app/build.gradle):
   ```gradle
   implementation 'org.tensorflow:tensorflow-lite:2.13.0'
   implementation 'org.tensorflow:tensorflow-lite-gpu:2.13.0'
   implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
   ```

2. **Add Model to Assets**:
   - Place `internvl_mobile.tflite` in `app/src/main/assets/`

3. **Basic Usage**:
   ```kotlin
   val vlModel = InternVLMobile(context)
   
   // Prepare your image
   val bitmap = // Your image loading code
   
   // Prepare your text tokens
   val inputIds = // Your tokenization code
   
   // Make prediction
   val result = vlModel.predict(bitmap, inputIds)
   result?.let {
       // Handle the result
       println("Prediction successful")
   }
   
   // Don't forget to close
   vlModel.close()
   ```

## Image Processing

```kotlin
fun preprocessImage(bitmap: Bitmap): Bitmap {
    return Bitmap.createScaledBitmap(bitmap, 224, 224, true)
}
```

## Text Processing

```kotlin
fun preprocessText(text: String): IntArray {
    // Implement your tokenization logic here
    // This depends on your specific tokenizer
    val tokens = tokenize(text)
    
    val paddedTokens = IntArray(512) { 0 }
    for (i in tokens.indices.take(512)) {
        paddedTokens[i] = tokens[i]
    }
    
    return paddedTokens
}
```

## Performance Tips

- Use GPU delegate for faster inference:
  ```kotlin
  val options = Interpreter.Options()
  options.addDelegate(GpuDelegate())
  interpreter = Interpreter(model, options)
  ```

- Use NNAPI delegate when available:
  ```kotlin
  val options = Interpreter.Options()
  options.addDelegate(NnApiDelegate())
  interpreter = Interpreter(model, options)
  ```

- Consider using multiple threads:
  ```kotlin
  val options = Interpreter.Options()
  options.setNumThreads(4)
  interpreter = Interpreter(model, options)
  ```

## Troubleshooting

- **Model loading fails**: Check that the .tflite file is in the assets folder
- **Inference errors**: Verify input shapes and types match model expectations
- **Performance issues**: Try GPU or NNAPI delegates, adjust thread count
