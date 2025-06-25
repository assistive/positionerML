package com.example.internvl

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class InternVLMobile(context: Context) {
    private var interpreter: Interpreter? = null
    
    init {
        try {
            val model = loadModelFile(context, "internvl_mobile.tflite")
            interpreter = Interpreter(model)
            println("Model loaded successfully")
        } catch (e: IOException) {
            println("Failed to load model: ${e.message}")
        }
    }
    
    private fun loadModelFile(context: Context, modelPath: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    fun predict(image: Bitmap, inputIds: IntArray): FloatArray? {
        val interpreter = this.interpreter ?: return null
        
        try {
            // Prepare image input
            val tensorImage = TensorImage.fromBitmap(image)
            val imageBuffer = tensorImage.buffer
            
            // Prepare text input
            val textBuffer = java.nio.IntBuffer.allocate(inputIds.size)
            textBuffer.put(inputIds)
            textBuffer.rewind()
            
            // Prepare output
            val outputShape = interpreter.getOutputTensor(0).shape()
            val output = Array(outputShape[0]) { FloatArray(outputShape[1]) }
            
            // Run inference
            val inputs = arrayOf(imageBuffer, textBuffer)
            val outputs = mapOf(0 to output)
            
            interpreter.runForMultipleInputsOutputs(inputs, outputs)
            
            return output[0]
            
        } catch (e: Exception) {
            println("Prediction failed: ${e.message}")
            return null
        }
    }
    
    fun close() {
        interpreter?.close()
        interpreter = null
    }
}
