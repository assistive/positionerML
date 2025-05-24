# TinyLlama Android Integration

This package contains everything needed to integrate TinyLlama into your Android application.

## Quick Start

### 1. Add to your project

Copy the TinyLlama.kt file to your project:
```
src/main/java/com/tinyllama/TinyLlama.kt
```

Add the TensorFlow Lite model to your assets:
```
src/main/assets/tinyllama_mobile.tflite
src/main/assets/tokenizer_info.json
```

### 2. Add dependencies to build.gradle

```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3'
    implementation 'com.google.code.gson:gson:2.10.1'
}
```

### 3. Basic Usage

```kotlin
class MainActivity : AppCompatActivity() {
    private lateinit var tinyLlama: TinyLlama
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Initialize TinyLlama
        tinyLlama = TinyLlama(this)
        
        // Generate text
        lifecycleScope.launch {
            val result = tinyLlama.generate(
                prompt = "Hello, how are you?",
                maxTokens = 50,
                temperature = 0.8f
            )
            // Use the generated text
            println(result)
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        tinyLlama.close() // Clean up resources
    }
}
```

### 4. Advanced Configuration

```kotlin
// Generate with custom parameters
val result = tinyLlama.generate(
    prompt = "Write a story about",
    maxTokens = 100,
    temperature = 1.0f,  // Higher = more creative
    topK = 40           // Top-K sampling
)

// Get model information
val modelInfo = tinyLlama.getModelInfo()
if (modelInfo.isLoaded) {
    println("Model loaded successfully")
    println("Vocab size: ${modelInfo.vocabSize}")
}
```

## Performance Tips

1. **Use GPU acceleration**: The library automatically tries to use GPU/NNAPI delegates
2. **Adjust temperature**: Lower values (0.1-0.5) for focused output, higher (0.8-1.2) for creativity
3. **Limit max tokens**: Fewer tokens = faster generation
4. **Keep model loaded**: Don't recreate TinyLlama instance frequently

## Model Files

- `tinyllama_mobile.tflite`: The quantized TensorFlow Lite model
- `tokenizer_info.json`: Tokenizer configuration and vocabulary info

## Requirements

- Android API 24+ (Android 7.0)
- ARM64 or x86_64 architecture recommended
- ~50MB+ available memory for model loading

## Troubleshooting

**Model not loading**: Check that .tflite file is in assets folder
**Slow performance**: Ensure GPU delegate is working or reduce model size
**Out of memory**: Try quantized model or reduce batch size

See the example app in the `examples/` folder for a complete implementation.
