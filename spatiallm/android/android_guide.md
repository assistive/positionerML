# SpatialLM Android Integration Guide

This guide provides step-by-step instructions for integrating the SpatialLM model into an Android application.

## Prerequisites

- Android Studio Arctic Fox (2020.3.1) or higher
- Android SDK 21 or higher
- Kotlin 1.5.31 or higher
- Gradle 7.0.2 or higher

## Project Setup

### 1. Add Dependencies

Add the following dependencies to your app's `build.gradle` file:

```gradle
dependencies {
    // TensorFlow Lite
    implementation 'org.tensorflow:tensorflow-lite:2.12.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.3'
    implementation 'org.tensorflow:tensorflow-lite-metadata:0.4.3'
    
    // Optional: GPU delegate
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.12.0'
    
    // Kotlin Coroutines
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.6.4'
}
```

### 2. Add TensorFlow Lite Model

1. Create an `assets` directory in your project if it doesn't exist:
   ```
   app/src/main/assets/
   ```

2. Copy the converted SpatialLM TFLite model (`spatialLM_model.tflite`) to the assets directory.

3. Copy the tokenizer vocabulary files to the assets directory.

### 3. Add Permissions

Add the following permissions to your `AndroidManifest.xml` file if needed:

```xml
<!-- For model downloading (if applicable) -->
<uses-permission android:name="android.permission.INTERNET" />
<!-- For model caching -->
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" 
                 android:maxSdkVersion="28" />
```

## Implementation

### 1. Create SpatialLM Tokenizer Class

Create a Kotlin class to handle tokenization:

```kotlin
class SpatialLMTokenizer(private val context: Context) {
    
    private val vocabMap: Map<String, Int>
    private val idToTokenMap: Map<Int, String>
    private val padToken: String
    private val eosToken: String
    
    init {
        // Load vocabulary from assets
        val vocabInputStream = context.assets.open("vocab.txt")
        val tokens = vocabInputStream.bufferedReader().readLines()
        
        // Create token to ID mapping
        vocabMap = tokens.mapIndexed { index, token -> token to index }.toMap()
        
        // Create ID to token mapping
        idToTokenMap = vocabMap.entries.associate { (k, v) -> v to k }
        
        // Load special tokens from model_metadata.json
        val metadataInputStream = context.assets.open("model_metadata.json")
        val metadataJson = JSONObject(metadataInputStream.bufferedReader().readText())
        val specialTokensJson = metadataJson.getJSONObject("special_tokens")
        
        padToken = specialTokensJson.getString("pad_token")
        eosToken = specialTokensJson.getString("eos_token")
    }
    
    fun tokenize(text: String, maxLength: Int = 64): Pair<IntArray, IntArray> {
        // Simple whitespace tokenization for demonstration
        // In a real implementation, you'd want to match the original tokenizer's behavior
        val tokens = text.split(" ")
            .flatMap { it.split("\n") }
            .filter { it.isNotEmpty() }
            .map { it.lowercase() }
        
        // Convert tokens to IDs
        val inputIds = mutableListOf<Int>()
        for (token in tokens) {
            val id = vocabMap[token] ?: vocabMap["<unk>"] ?: 0
            inputIds.add(id)
            
            if (inputIds.size >= maxLength - 1) {
                break
            }
        }
        
        // Add EOS token
        val eosId = vocabMap[eosToken] ?: 0
        inputIds.add(eosId)
        
        // Create attention mask (1 for real tokens, 0 for padding)
        val attentionMask = IntArray(maxLength) { 1 }
        
        // Pad input IDs if necessary
        val paddedInputIds = IntArray(maxLength) { vocabMap[padToken] ?: 0 }
        inputIds.forEachIndexed { index, id ->
            if (index < maxLength) {
                paddedInputIds[index] = id
            }
        }
        
        return Pair(paddedInputIds, attentionMask)
    }
    
    fun decode(ids: IntArray): String {
        return ids.map { idToTokenMap[it] ?: "" }
            .filter { it.isNotEmpty() && it != padToken && it != eosToken }
            .joinToString(" ")
    }
}