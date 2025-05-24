package com.tinyllama.example

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
