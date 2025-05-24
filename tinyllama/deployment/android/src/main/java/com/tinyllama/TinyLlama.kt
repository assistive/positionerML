package com.tinyllama

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
            .replace(Regex("[^a-zA-Z0-9\s]"), " ")
            .split(Regex("\s+"))
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
