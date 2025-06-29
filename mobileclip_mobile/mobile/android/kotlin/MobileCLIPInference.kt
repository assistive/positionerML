/**
 * MobileCLIPInference.kt
 * MobileCLIP Android Integration
 * 
 * Provides easy-to-use interface for MobileCLIP inference on Android
 */
package com.mobileclip.inference

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.*
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.sqrt

class MobileCLIPInference(private val context: Context) {
    
    companion object {
        private const val TAG = "MobileCLIPInference"
        private const val IMAGE_MODEL_NAME = "mobileclip_image.tflite"
        private const val TEXT_MODEL_NAME = "mobileclip_text.tflite"
        private const val IMAGE_SIZE = 224
        private const val TEXT_MAX_LENGTH = 77
        private const val EMBEDDING_DIM = 512
    }
    
    private var imageInterpreter: Interpreter? = null
    private var textInterpreter: Interpreter? = null
    private var isInitialized = false
    
    private val imageProcessor = ImageProcessor.Builder()
        .add(ResizeOp(IMAGE_SIZE, IMAGE_SIZE, ResizeOp.ResizeMethod.BILINEAR))
        .build()
    
    /**
     * Initialize the MobileCLIP inference engines
     */
    suspend fun initialize(): Boolean = withContext(Dispatchers.IO) {
        try {
            // Load models from assets
            val imageModelBuffer = loadModelFile(IMAGE_MODEL_NAME)
            val textModelBuffer = loadModelFile(TEXT_MODEL_NAME)
            
            // Create interpreters
            val options = Interpreter.Options().apply {
                setNumThreads(4)
                setUseNNAPI(true)
                setUseXNNPACK(true)
            }
            
            imageInterpreter = Interpreter(imageModelBuffer, options)
            textInterpreter = Interpreter(textModelBuffer, options)
            
            isInitialized = true
            Log.d(TAG, "MobileCLIP models initialized successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize models: ${e.message}")
            false
        }
    }
    
    /**
     * Extract image features from bitmap
     */
    suspend fun extractImageFeatures(bitmap: Bitmap): FloatArray? = withContext(Dispatchers.Default) {
        if (!isInitialized || imageInterpreter == null) {
            Log.e(TAG, "Models not initialized")
            return@withContext null
        }
        
        try {
            // Preprocess image
            val tensorImage = TensorImage.fromBitmap(bitmap)
            val processedImage = imageProcessor.process(tensorImage)
            
            // Prepare input buffer
            val inputBuffer = processedImage.buffer
            
            // Prepare output buffer
            val outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, EMBEDDING_DIM), org.tensorflow.lite.DataType.FLOAT32)
            
            // Run inference
            imageInterpreter?.run(inputBuffer, outputBuffer.buffer)
            
            // Extract features
            val features = outputBuffer.floatArray
            
            // Normalize features
            normalizeVector(features)
            
        } catch (e: Exception) {
            Log.e(TAG, "Image feature extraction failed: ${e.message}")
            null
        }
    }
    
    /**
     * Extract text features from string
     */
    suspend fun extractTextFeatures(text: String): FloatArray? = withContext(Dispatchers.Default) {
        if (!isInitialized || textInterpreter == null) {
            Log.e(TAG, "Models not initialized")
            return@withContext null
        }
        
        try {
            // Tokenize text
            val tokens = tokenizeText(text)
            
            // Prepare input buffer
            val inputBuffer = ByteBuffer.allocateDirect(TEXT_MAX_LENGTH * 4)
            inputBuffer.order(ByteOrder.nativeOrder())
            
            for (token in tokens) {
                inputBuffer.putInt(token)
            }
            inputBuffer.rewind()
            
            // Prepare output buffer
            val outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, EMBEDDING_DIM), org.tensorflow.lite.DataType.FLOAT32)
            
            // Run inference
            textInterpreter?.run(inputBuffer, outputBuffer.buffer)
            
            // Extract features
            val features = outputBuffer.floatArray
            
            // Normalize features
            normalizeVector(features)
            
        } catch (e: Exception) {
            Log.e(TAG, "Text feature extraction failed: ${e.message}")
            null
        }
    }
    
    /**
     * Compute similarity between image and text features
     */
    fun computeSimilarity(imageFeatures: FloatArray, textFeatures: FloatArray): Float {
        if (imageFeatures.size != textFeatures.size) {
            Log.e(TAG, "Feature vector sizes don't match")
            return 0f
        }
        
        var dotProduct = 0f
        for (i in imageFeatures.indices) {
            dotProduct += imageFeatures[i] * textFeatures[i]
        }
        
        return dotProduct
    }
    
    /**
     * Perform zero-shot classification
     */
    suspend fun zeroShotClassify(bitmap: Bitmap, labels: List<String>): List<Pair<String, Float>>? {
        val imageFeatures = extractImageFeatures(bitmap) ?: return null
        
        val results = mutableListOf<Pair<String, Float>>()
        
        for (label in labels) {
            val textFeatures = extractTextFeatures(label) ?: continue
            val similarity = computeSimilarity(imageFeatures, textFeatures)
            results.add(Pair(label, similarity))
        }
        
        // Sort by confidence (similarity score)
        return results.sortedByDescending { it.second }
    }
    
    /**
     * Simple text tokenization (simplified implementation)
     */
    private fun tokenizeText(text: String): IntArray {
        val tokens = IntArray(TEXT_MAX_LENGTH) { 0 }
        
        // Simple tokenization - in practice, use proper tokenizer
        val words = text.lowercase().split("\\s+".toRegex())
        
        tokens[0] = 49406 // Start token
        
        for (i in words.indices) {
            if (i < TEXT_MAX_LENGTH - 2) {
                tokens[i + 1] = words[i].hashCode() % 30000 // Simplified hashing
            }
        }
        
        if (words.size < TEXT_MAX_LENGTH - 2) {
            tokens[words.size + 1] = 49407 // End token
        }
        
        return tokens
    }
    
    /**
     * Normalize feature vector to unit length
     */
    private fun normalizeVector(vector: FloatArray) {
        var norm = 0f
        for (value in vector) {
            norm += value * value
        }
        norm = sqrt(norm)
        
        if (norm > 0f) {
            for (i in vector.indices) {
                vector[i] /= norm
            }
        }
    }
    
    /**
     * Load model file from assets
     */
    private fun loadModelFile(fileName: String): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(fileName)
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    /**
     * Clean up resources
     */
    fun close() {
        imageInterpreter?.close()
        textInterpreter?.close()
        imageInterpreter = null
        textInterpreter = null
        isInitialized = false
        Log.d(TAG, "MobileCLIP inference closed")
    }
}

/**
 * Usage example:
 * 
 * class MainActivity : AppCompatActivity() {
 *     private lateinit var mobileCLIP: MobileCLIPInference
 *     
 *     override fun onCreate(savedInstanceState: Bundle?) {
 *         super.onCreate(savedInstanceState)
 *         
 *         mobileCLIP = MobileCLIPInference(this)
 *         
 *         lifecycleScope.launch {
 *             if (mobileCLIP.initialize()) {
 *                 // Ready to use
 *                 val results = mobileCLIP.zeroShotClassify(bitmap, listOf("dog", "cat", "bird"))
 *                 results?.forEach { (label, confidence) ->
 *                     Log.d("Results", "$label: $confidence")
 *                 }
 *             }
 *         }
 *     }
 *     
 *     override fun onDestroy() {
 *         super.onDestroy()
 *         mobileCLIP.close()
 *     }
 * }
 */
