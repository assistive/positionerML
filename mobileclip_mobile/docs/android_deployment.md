# Android Deployment Guide

Complete guide for deploying MobileCLIP models on Android devices.

## Prerequisites

- Android Studio
- Android API 24+ (Android 7.0)
- TensorFlow Lite 2.13.0+
- Device with NNAPI support (recommended)

## Model Conversion

Convert your MobileCLIP model to TensorFlow Lite format:

```bash
python scripts/convert/convert_models.py --model mobileclip_s0 --platforms android
```

This creates TensorFlow Lite models in `models/converted/`:
- `mobileclip_s0.tflite` - Complete model

## Android Studio Integration

### 1. Add Dependencies

Add to your app-level `build.gradle`:

```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.6.4'
}
```

### 2. Add Model to Assets

1. Create `assets` folder in `app/src/main/` if it doesn't exist
2. Copy `.tflite` files to `app/src/main/assets/`

### 3. Add Kotlin Integration Class

Add `MobileCLIPInference.kt` to your project.

### 4. Basic Usage

```kotlin
class MainActivity : AppCompatActivity() {
    private lateinit var mobileCLIP: MobileCLIPInference
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        mobileCLIP = MobileCLIPInference(this)
        
        lifecycleScope.launch {
            if (mobileCLIP.initialize()) {
                Log.d(TAG, "MobileCLIP initialized successfully")
            } else {
                Log.e(TAG, "Failed to initialize MobileCLIP")
            }
        }
    }
    
    private fun classifyImage(bitmap: Bitmap) {
        lifecycleScope.launch {
            val labels = listOf("dog", "cat", "bird", "car", "airplane")
            val results = mobileCLIP.zeroShotClassify(bitmap, labels)
            
            results?.take(3)?.forEach { (label, confidence) ->
                Log.d(TAG, "$label: ${String.format("%.2f", confidence * 100)}%")
            }
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        mobileCLIP.close()
    }
}
```

## Performance Optimization

### NNAPI Acceleration

Enable NNAPI for faster inference:

```kotlin
val options = Interpreter.Options().apply {
    setUseNNAPI(true)
    setNumThreads(4)
}
```

### GPU Delegate

Use GPU acceleration when available:

```kotlin
val gpuDelegate = GpuDelegate()
val options = Interpreter.Options().apply {
    addDelegate(gpuDelegate)
}
```

### Memory Optimization

```kotlin
// Process images on background thread
lifecycleScope.launch(Dispatchers.Default) {
    val results = mobileCLIP.extractImageFeatures(bitmap)
    
    withContext(Dispatchers.Main) {
        // Update UI
    }
}
```

## Camera Integration

### Basic Camera Capture

```kotlin
private fun setupCamera() {
    val imageCapture = ImageCapture.Builder().build()
    
    imageCapture.takePicture(
        outputFileOptions,
        ContextCompat.getMainExecutor(this),
        object : ImageCapture.OnImageSavedCallback {
            override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                val bitmap = BitmapFactory.decodeFile(output.savedUri?.path)
                classifyImage(bitmap)
            }
            
            override fun onError(exception: ImageCaptureException) {
                Log.e(TAG, "Image capture failed: ${exception.message}")
            }
        }
    )
}
```

## Troubleshooting

### Common Issues

1. **Model loading fails:**
   - Verify `.tflite` files are in `assets` folder
   - Check file names match code expectations
   - Ensure sufficient storage space

2. **Slow inference:**
   - Enable NNAPI acceleration
   - Use GPU delegate if available
   - Check model quantization

3. **Out of memory:**
   - Resize large images before processing
   - Process images on background thread
   - Close unused model instances

### Performance Tips

- Enable hardware acceleration (NNAPI/GPU)
- Use quantized models for faster inference
- Resize images to 224x224 before processing
- Batch process multiple images when possible

## Example App

See `examples/android/MobileCLIPDemo/` for a complete example application.
