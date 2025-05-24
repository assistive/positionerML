#!/usr/bin/env python3
"""
Android Deployment Script for TinyLlama
Creates deployment package with TensorFlow Lite model and Kotlin integration code.
"""

import os
import shutil
from pathlib import Path

def create_android_deployment_package(model_dir: str, output_dir: str = "deployment/android"):
    """Create Android deployment package."""
    
    print("Creating Android deployment package...")
    
    # Create directory structure
    android_dir = Path(output_dir)
    (android_dir / "src" / "main" / "assets").mkdir(parents=True, exist_ok=True)
    (android_dir / "src" / "main" / "java" / "com" / "tinyllama").mkdir(parents=True, exist_ok=True)
    (android_dir / "src" / "main" / "res" / "layout").mkdir(parents=True, exist_ok=True)
    (android_dir / "src" / "main" / "res" / "values").mkdir(parents=True, exist_ok=True)
    (android_dir / "examples").mkdir(parents=True, exist_ok=True)
    
    # Copy TFLite model
    model_files = list(Path(model_dir).glob("*.tflite"))
    if model_files:
        shutil.copy2(model_files[0], android_dir / "src" / "main" / "assets")
        print(f"Copied TFLite model: {model_files[0].name}")
    
    # Copy tokenizer info
    tokenizer_file = Path(model_dir) / "tokenizer_info.json"
    if tokenizer_file.exists():
        shutil.copy2(tokenizer_file, android_dir / "src" / "main" / "assets")
    
    # Create build.gradle
    build_gradle = '''plugins {
    id 'com.android.library'
    id 'kotlin-android'
}

android {
    compileSdk 34

    defaultConfig {
        minSdk 24
        targetSdk 34
        
        testInstrumentationRunner "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            minifyEnabled false
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
        }
    }
    
    compileOptions {
        sourceCompatibility JavaVersion.VERSION_1_8
        targetCompatibility JavaVersion.VERSION_1_8
    }
    
    kotlinOptions {
        jvmTarget = '1.8'
    }
}

dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
    implementation 'androidx.core:core-ktx:1.12.0'
    implementation 'androidx.appcompat:appcompat:1.6.1'
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3'
    implementation 'com.google.code.gson:gson:2.10.1'
    implementation 'androidx.lifecycle:lifecycle-runtime-ktx:2.7.0'
    
    testImplementation 'junit:junit:4.13.2'
    androidTestImplementation 'androidx.test.ext:junit:1.1.5'
    androidTestImplementation 'androidx.test.espresso:espresso-core:3.5.1'
}
'''
    
    with open(android_dir / "build.gradle", "w") as f:
        f.write(build_gradle)
    
    # Create Kotlin implementation
    kotlin_code = '''package com.tinyllama

import android.content.Context
import android.content.res.AssetManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import com.google.gson.Gson
import java.io.FileInputStream
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.random.Random
import kotlin.math.exp

class TinyLlama(private val context: Context) {
    
    private var interpreter: Interpreter? = null
    private var tokenizer: TinyLlamaTokenizer? = null
    private val maxLength = 256
    
    init {
        loadModel()
        loadTokenizer()
    }
    
    private fun loadModel() {
        try {
            val modelBuffer = loadModelFile("tinyllama_mobile.tflite")
            val options = Interpreter.Options().apply {
                setNumThreads(4)
                setUseXNNPACK(true)
                
                // Try to use GPU delegate if available
                try {
                    addDelegate(org.tensorflow.lite.gpu.GpuDelegate())
                    println("Using GPU delegate")
                } catch (e: Exception) {
                    println("GPU delegate not available, using CPU: ${e.message}")
                }
                
                // Try to use NNAPI delegate if available
                try {
                    addDelegate(org.tensorflow.lite.nnapi.NnApiDelegate())
                    println("Using NNAPI delegate")
                } catch (e: Exception) {
                    println("NNAPI delegate not available: ${e.message}")
                }
            }
            
            interpreter = Interpreter(modelBuffer, options)
            println("TinyLlama model loaded successfully")
            
            // Print input/output tensor info
            val inputTensor = interpreter!!.getInputTensor(0)
            val outputTensor = interpreter!!.getOutputTensor(0)
            println("Input tensor shape: ${inputTensor.shape().contentToString()}")
            println("Output tensor shape: ${outputTensor.shape().contentToString()}")
            
        } catch (e: Exception) {
            println("Error loading model: ${e.message}")
            e.printStackTrace()
        }
    }
    
    private fun loadTokenizer() {
        try {
            val tokenizerJson = context.assets.open("tokenizer_info.json")
                .bufferedReader().use { it.readText() }
            val tokenizerInfo = Gson().fromJson(tokenizerJson, TokenizerInfo::class.java)
            tokenizer = TinyLlamaTokenizer(tokenizerInfo)
            println("Tokenizer loaded successfully")
        } catch (e: Exception) {
            println("Error loading tokenizer: ${e.message}")
            // Create default tokenizer if file not found
            val defaultInfo = TokenizerInfo(
                vocabSize = 32000,
                padTokenId = 0,
                eosTokenId = 2,
                bosTokenId = 1,
                maxLength = 256
            )
            tokenizer = TinyLlamaTokenizer(defaultInfo)
        }
    }
    
    private fun loadModelFile(fileName: String): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(fileName)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    suspend fun generate(
        prompt: String, 
        maxTokens: Int = 50,
        temperature: Float = 0.8f,
        topK: Int = 40
    ): String = withContext(Dispatchers.Default) {
        val interpreter = this@TinyLlama.interpreter ?: return@withContext "Model not loaded"
        val tokenizer = this@TinyLlama.tokenizer ?: return@withContext "Tokenizer not loaded"
        
        try {
            // Tokenize input
            val tokens = tokenizer.encode(prompt).toMutableList()
            val originalLength = tokens.size
            
            // Generate tokens
            repeat(maxTokens) {
                if (tokens.size >= maxLength - 1) return@repeat
                
                // Prepare input - pad with zeros
                val inputArray = Array(1) { IntArray(maxLength) { 0 } }
                tokens.forEachIndexed { index, token ->
                    if (index < maxLength) {
                        inputArray[0][index] = token
                    }
                }
                
                // Run inference
                val vocabSize = tokenizer.vocabSize
                val outputArray = Array(1) { Array(maxLength) { FloatArray(vocabSize) } }
                
                interpreter.run(inputArray, outputArray)
                
                // Get logits for the last token position
                val lastPosition = minOf(tokens.size - 1, maxLength - 1)
                val logits = outputArray[0][lastPosition]
                
                // Sample next token
                val nextToken = sampleToken(logits, temperature, topK)
                tokens.add(nextToken)
                
                // Stop if EOS token
                if (nextToken == tokenizer.eosTokenId) {
                    break
                }
            }
            
            // Decode only the generated part
            val generatedTokens = tokens.subList(originalLength, tokens.size)
            return@withContext tokenizer.decode(generatedTokens)
            
        } catch (e: Exception) {
            println("Error during generation: ${e.message}")
            e.printStackTrace()
            return@withContext "Error during generation: ${e.message}"
        }
    }
    
    private fun sampleToken(logits: FloatArray, temperature: Float, topK: Int): Int {
        if (temperature <= 0f) {
            // Greedy sampling
            return logits.indices.maxByOrNull { logits[it] } ?: 0
        }
        
        // Apply temperature
        val scaledLogits = logits.map { it / temperature }.toFloatArray()
        
        // Top-K sampling
        val topKIndices = scaledLogits.indices
            .sortedByDescending { scaledLogits[it] }
            .take(topK)
        
        // Convert to probabilities using softmax
        val maxLogit = topKIndices.maxOfOrNull { scaledLogits[it] } ?: 0f
        val expValues = topKIndices.map { 
            exp(scaledLogits[it] - maxLogit) 
        }
        val sumExp = expValues.sum()
        val probabilities = expValues.map { it / sumExp }
        
        // Sample from the distribution
        val random = Random.nextFloat()
        var cumulativeProb = 0f
        
        for (i in probabilities.indices) {
            cumulativeProb += probabilities[i]
            if (random <= cumulativeProb) {
                return topKIndices[i]
            }
        }
        
        return topKIndices.last()
    }
    
    fun getModelInfo(): ModelInfo {
        val interpreter = this.interpreter
        val tokenizer = this.tokenizer
        
        return if (interpreter != null && tokenizer != null) {
            val inputTensor = interpreter.getInputTensor(0)
            val outputTensor = interpreter.getOutputTensor(0)
            
            ModelInfo(
                isLoaded = true,
                inputShape = inputTensor.shape(),
                outputShape = outputTensor.shape(),
                vocabSize = tokenizer.vocabSize,
                maxLength = maxLength
            )
        } else {
            ModelInfo(isLoaded = false)
        }
    }
    
    fun close() {
        interpreter?.close()
        interpreter = null
        println("TinyLlama resources cleaned up")
    }
}

class TinyLlamaTokenizer(private val tokenizerInfo: TokenizerInfo) {
    val vocabSize: Int = tokenizerInfo.vocabSize
    val eosTokenId: Int = tokenizerInfo.eosTokenId ?: 2
    private val bosTokenId: Int = tokenizerInfo.bosTokenId ?: 1
    private val padTokenId: Int = tokenizerInfo.padTokenId ?: 0
    
    // Simple vocabulary for demonstration
    private val specialTokens = mapOf(
        "<pad>" to padTokenId,
        "<s>" to bosTokenId, 
        "</s>" to eosTokenId,
        "<unk>" to 3
    )
    
    fun encode(text: String): List<Int> {
        // Add BOS token
        val tokens = mutableListOf(bosTokenId)
        
        // Simple word-level tokenization (in practice, use SentencePiece)
        val words = text.lowercase()
            .replace(Regex("[^a-zA-Z0-9\\s]"), " ")
            .split(Regex("\\s+"))
            .filter { it.isNotEmpty() }
        
        // Convert words to token IDs
        for (word in words) {
            val tokenId = when {
                word in specialTokens -> specialTokens[word]!!
                else -> {
                    // Simple hash-based mapping for demonstration
                    val hash = Math.abs(word.hashCode()) % (vocabSize - 100) + 100
                    hash
                }
            }
            tokens.add(tokenId)
        }
        
        return tokens
    }
    
    fun decode(tokens: List<Int>): String {
        val reverseSpecialTokens = specialTokens.entries.associateBy({ it.value }) { it.key }
        
        val words = tokens.mapNotNull { tokenId ->
            when (tokenId) {
                padTokenId -> null // Skip padding tokens
                bosTokenId -> null // Skip BOS token  
                eosTokenId -> null // Skip EOS token
                in reverseSpecialTokens -> reverseSpecialTokens[tokenId]
                else -> "word_$tokenId" // Placeholder for actual vocab lookup
            }
        }
        
        return words.joinToString(" ")
    }
}

data class TokenizerInfo(
    val vocabSize: Int,
    val padTokenId: Int?,
    val eosTokenId: Int?,
    val bosTokenId: Int?,
    val maxLength: Int
)

data class ModelInfo(
    val isLoaded: Boolean,
    val inputShape: IntArray? = null,
    val outputShape: IntArray? = null,
    val vocabSize: Int = 0,
    val maxLength: Int = 0
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as ModelInfo

        if (isLoaded != other.isLoaded) return false
        if (inputShape != null) {
            if (other.inputShape == null) return false
            if (!inputShape.contentEquals(other.inputShape)) return false
        } else if (other.inputShape != null) return false
        if (outputShape != null) {
            if (other.outputShape == null) return false
            if (!outputShape.contentEquals(other.outputShape)) return false
        } else if (other.outputShape != null) return false
        if (vocabSize != other.vocabSize) return false
        if (maxLength != other.maxLength) return false

        return true
    }

    override fun hashCode(): Int {
        var result = isLoaded.hashCode()
        result = 31 * result + (inputShape?.contentHashCode() ?: 0)
        result = 31 * result + (outputShape?.contentHashCode() ?: 0)
        result = 31 * result + vocabSize
        result = 31 * result + maxLength
        return result
    }
}
'''
    
    with open(android_dir / "src" / "main" / "java" / "com" / "tinyllama" / "TinyLlama.kt", "w") as f:
        f.write(kotlin_code)
    
    # Create example usage activity
    example_code = '''package com.tinyllama.example

import android.os.Bundle
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.tinyllama.TinyLlama
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {
    
    private lateinit var tinyLlama: TinyLlama
    private lateinit var promptEditText: EditText
    private lateinit var generateButton: Button
    private lateinit var resultTextView: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var modelInfoTextView: TextView
    private lateinit var temperatureSeekBar: SeekBar
    private lateinit var temperatureLabel: TextView
    private lateinit var maxTokensEditText: EditText
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        initViews()
        initTinyLlama()
        setupClickListeners()
        displayModelInfo()
    }
    
    private fun initViews() {
        promptEditText = findViewById(R.id.promptEditText)
        generateButton = findViewById(R.id.generateButton)
        resultTextView = findViewById(R.id.resultTextView)
        progressBar = findViewById(R.id.progressBar)
        modelInfoTextView = findViewById(R.id.modelInfoTextView)
        temperatureSeekBar = findViewById(R.id.temperatureSeekBar)
        temperatureLabel = findViewById(R.id.temperatureLabel)
        maxTokensEditText = findViewById(R.id.maxTokensEditText)
        
        // Set default values
        maxTokensEditText.setText("50")
        temperatureSeekBar.max = 200 // 0.0 to 2.0, scaled by 100
        temperatureSeekBar.progress = 80 // 0.8 default
        updateTemperatureLabel(0.8f)
    }
    
    private fun initTinyLlama() {
        tinyLlama = TinyLlama(this)
    }
    
    private fun setupClickListeners() {
        generateButton.setOnClickListener {
            val prompt = promptEditText.text.toString().trim()
            if (prompt.isNotEmpty()) {
                generateText(prompt)
            } else {
                Toast.makeText(this, "Please enter a prompt", Toast.LENGTH_SHORT).show()
            }
        }
        
        temperatureSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                val temperature = progress / 100f
                updateTemperatureLabel(temperature)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
    }
    
    private fun updateTemperatureLabel(temperature: Float) {
        temperatureLabel.text = String.format("Temperature: %.2f", temperature)
    }
    
    private fun displayModelInfo() {
        val modelInfo = tinyLlama.getModelInfo()
        val infoText = if (modelInfo.isLoaded) {
            """Model Status: Loaded ✓
Vocabulary Size: ${modelInfo.vocabSize}
Max Length: ${modelInfo.maxLength}
Input Shape: ${modelInfo.inputShape?.contentToString() ?: "N/A"}
Output Shape: ${modelInfo.outputShape?.contentToString() ?: "N/A"}"""
        } else {
            "Model Status: Not Loaded ✗"
        }
        
        modelInfoTextView.text = infoText
    }
    
    private fun generateText(prompt: String) {
        generateButton.isEnabled = false
        progressBar.visibility = ProgressBar.VISIBLE
        resultTextView.text = "Generating..."
        
        val temperature = temperatureSeekBar.progress / 100f
        val maxTokens = maxTokensEditText.text.toString().toIntOrNull() ?: 50
        
        lifecycleScope.launch {
            try {
                val startTime = System.currentTimeMillis()
                val result = tinyLlama.generate(
                    prompt = prompt, 
                    maxTokens = maxTokens,
                    temperature = temperature
                )
                val endTime = System.currentTimeMillis()
                val duration = endTime - startTime
                
                val fullResult = """Generated Text:
$result

---
Generation Time: ${duration}ms
Temperature: $temperature
Max Tokens: $maxTokens"""
                
                resultTextView.text = fullResult
                
            } catch (e: Exception) {
                resultTextView.text = "Error: ${e.message}"
                e.printStackTrace()
            } finally {
                generateButton.isEnabled = true
                progressBar.visibility = ProgressBar.GONE
            }
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        tinyLlama.close()
    }
}
'''
    
    with open(android_dir / "examples" / "MainActivity.kt", "w") as f:
        f.write(example_code)
    
    # Create layout file
    layout_xml = '''<?xml version="1.0" encoding="utf-8"?>
<ScrollView xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:padding="16dp">

    <LinearLayout
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:orientation="vertical">

        <TextView
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:text="TinyLlama Mobile Demo"
            android:textSize="24sp"
            android:textStyle="bold"
            android:layout_marginBottom="16dp"
            android:layout_gravity="center_horizontal" />

        <TextView
            android:id="@+id/modelInfoTextView"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:text="Loading model info..."
            android:textSize="12sp"
            android:fontFamily="monospace"
            android:background="#f0f0f0"
            android:padding="8dp"
            android:layout_marginBottom="16dp" />

        <TextView
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:text="Enter your prompt:"
            android:textSize="16sp"
            android:textStyle="bold"
            android:layout_marginBottom="8dp" />

        <EditText
            android:id="@+id/promptEditText"
            android:layout_width="match_parent"
            android:layout_height="100dp"
            android:gravity="top|start"
            android:hint="Type your prompt here..."
            android:inputType="textMultiLine"
            android:background="@android:drawable/edit_text"
            android:padding="8dp"
            android:layout_marginBottom="16dp" />

        <LinearLayout
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:orientation="horizontal"
            android:layout_marginBottom="16dp">

            <TextView
                android:layout_width="wrap_content"
                android:layout_height="wrap_content"
                android:text="Max Tokens:"
                android:layout_gravity="center_vertical"
                android:layout_marginEnd="8dp" />

            <EditText
                android:id="@+id/maxTokensEditText"
                android:layout_width="80dp"
                android:layout_height="wrap_content"
                android:inputType="number"
                android:text="50"
                android:layout_marginEnd="16dp" />

        </LinearLayout>

        <TextView
            android:id="@+id/temperatureLabel"
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:text="Temperature: 0.80"
            android:layout_marginBottom="8dp" />

        <SeekBar
            android:id="@+id/temperatureSeekBar"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:layout_marginBottom="16dp" />

        <Button
            android:id="@+id/generateButton"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:text="Generate Text"
            android:textSize="16sp"
            android:layout_marginBottom="16dp" />

        <ProgressBar
            android:id="@+id/progressBar"
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:layout_gravity="center_horizontal"
            android:visibility="gone"
            android:layout_marginBottom="16dp" />

        <TextView
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:text="Generated Text:"
            android:textSize="16sp"
            android:textStyle="bold"
            android:layout_marginBottom="8dp" />

        <TextView
            android:id="@+id/resultTextView"
            android:layout_width="match_parent"
            android:layout_height="200dp"
            android:text="Generated text will appear here..."
            android:background="#f9f9f9"
            android:padding="12dp"
            android:textIsSelectable="true"
            android:scrollbars="vertical"
            android:fontFamily="monospace"
            android:textSize="14sp"
            android:gravity="top|start" />

    </LinearLayout>

</ScrollView>
'''
    
    with open(android_dir / "src" / "main" / "res" / "layout" / "activity_main.xml", "w") as f:
        f.write(layout_xml)
    
    # Create strings.xml
    strings_xml = '''<?xml version="1.0" encoding="utf-8"?>
<resources>
    <string name="app_name">TinyLlama Mobile</string>
    <string name="prompt_hint">Enter your prompt here...</string>
    <string name="generate_button">Generate Text</string>
    <string name="generating">Generating...</string>
    <string name="model_not_loaded">Model not loaded</string>
    <string name="enter_prompt">Please enter a prompt</string>
</resources>
'''
    
    with open(android_dir / "src" / "main" / "res" / "values" / "strings.xml", "w") as f:
        f.write(strings_xml)
    
    # Create AndroidManifest.xml for the example
    manifest_xml = '''<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.tinyllama.example">

    <application
        android:allowBackup="true"
        android:icon="@mipmap/ic_launcher"
        android:label="@string/app_name"
        android:theme="@style/Theme.AppCompat.Light.DarkActionBar">
        
        <activity
            android:name=".MainActivity"
            android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
        
    </application>
    
</manifest>
'''
    
    with open(android_dir / "examples" / "AndroidManifest.xml", "w") as f:
        f.write(manifest_xml)
    
    # Create README with integration instructions
    readme_content = '''# TinyLlama Android Integration

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
'''
    
    with open(android_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    print(f"Android deployment package created at: {android_dir}")
    print("Contents:")
    print("- TinyLlama.kt: Main library implementation")
    print("- MainActivity.kt: Example usage")
    print("- build.gradle: Dependencies and build configuration")
    print("- README.md: Integration instructions")
    
    return str(android_dir)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python deploy_android.py <model_directory>")
        print("Example: python deploy_android.py ../models/mobile/android/")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    if not os.path.exists(model_dir):
        print(f"Error: Model directory {model_dir} does not exist")
        sys.exit(1)
        
    create_android_deployment_package(model_dir)
    print("\nAndroid deployment package created successfully!")
    print("You can now integrate TinyLlama into your Android project.")
