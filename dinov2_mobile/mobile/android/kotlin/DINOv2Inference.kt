package com.example.dinov2mobile

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class DINOv2Inference(private val context: Context) {
    
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var nnApiDelegate: NnApiDelegate? = null
    
    companion object {
        private const val TAG = "DINOv2Inference"
        private const val MODEL_FILE = "dinov2_mobile.tflite"
        private const val INPUT_SIZE = 224
        private const val FEATURE_SIZE = 384 // For ViT-S
    }
    
    fun initialize(): Boolean {
        return try {
            val model = loadModelFile()
            val options = Interpreter.Options()
            
            // GPU acceleration
            val compatibilityList = CompatibilityList()
            if (compatibilityList.isDelegateSupportedOnThisDevice) {
                gpuDelegate = GpuDelegate()
                options.addDelegate(gpuDelegate)
                Log.d(TAG, "GPU delegate added")
            }
            
            // NNAPI acceleration
            nnApiDelegate = NnApiDelegate()
            options.addDelegate(nnApiDelegate)
            
            interpreter = Interpreter(model, options)
            Log.d(TAG, "DINOv2 model initialized successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize model: ${e.message}")
            false
        }
    }
    
    fun extractFeatures(bitmap: Bitmap): FloatArray? {
        val interpreter = this.interpreter ?: return null
        
        try {
            // Preprocess image
            val inputBuffer = preprocessImage(bitmap)
            
            // Prepare output buffer
            val outputBuffer = ByteBuffer.allocateDirect(4 * FEATURE_SIZE)
            outputBuffer.order(ByteOrder.nativeOrder())
            
            // Run inference
            val startTime = System.currentTimeMillis()
            interpreter.run(inputBuffer, outputBuffer)
            val inferenceTime = System.currentTimeMillis() - startTime
            
            Log.d(TAG, "Inference time: ${inferenceTime}ms")
            
            // Convert output to FloatArray
            outputBuffer.rewind()
            val features = FloatArray(FEATURE_SIZE)
            outputBuffer.asFloatBuffer().get(features)
            
            return features
            
        } catch (e: Exception) {
            Log.e(TAG, "Feature extraction failed: ${e.message}")
            return null
        }
    }
    
    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        
        val inputBuffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3)
        inputBuffer.order(ByteOrder.nativeOrder())
        
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resizedBitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        
        // Normalize pixels (ImageNet normalization)
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)
        
        for (pixel in pixels) {
            val r = ((pixel shr 16 and 0xFF) / 255.0f - mean[0]) / std[0]
            val g = ((pixel shr 8 and 0xFF) / 255.0f - mean[1]) / std[1]
            val b = ((pixel and 0xFF) / 255.0f - mean[2]) / std[2]
            
            inputBuffer.putFloat(r)
            inputBuffer.putFloat(g)
            inputBuffer.putFloat(b)
        }
        
        return inputBuffer
    }
    
    private fun loadModelFile(): ByteBuffer {
        val assetFileDescriptor = context.assets.openFd(MODEL_FILE)
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    fun close() {
        interpreter?.close()
        gpuDelegate?.close()
        nnApiDelegate?.close()
    }
}
