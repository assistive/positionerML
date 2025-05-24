# bert_mobile/scripts/deploy.py

#!/usr/bin/env python3

import argparse
import sys
import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from mobile_converter import BERTMobileConverter

class BERTDeploymentManager:
    """Manage BERT model deployment to mobile platforms."""
    
    def __init__(self, config_path: str = "config/mobile_config.yaml"):
        self.converter = BERTMobileConverter(config_path)
        self.config = self.converter.config
    
    def create_ios_deployment_package(self, model_dir: str, output_dir: str) -> str:
        """Create iOS deployment package."""
        print("Creating iOS deployment package...")
        
        ios_package_dir = Path(output_dir) / "ios_deployment"
        ios_package_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model files
        model_files = list(Path(model_dir).glob("*.mlmodel"))
        if not model_files:
            raise ValueError("No CoreML model files found in model directory")
        
        for model_file in model_files:
            shutil.copy2(model_file, ios_package_dir)
            print(f"Copied model: {model_file.name}")
        
        # Copy tokenizer files
        tokenizer_dir = Path(model_dir) / "tokenizer"
        if tokenizer_dir.exists():
            shutil.copytree(tokenizer_dir, ios_package_dir / "tokenizer", dirs_exist_ok=True)
        
        # Create Swift integration code
        self.create_ios_integration_code(ios_package_dir)
        
        # Create Xcode project template
        self.create_ios_project_template(ios_package_dir)
        
        # Create documentation
        self.create_ios_documentation(ios_package_dir)
        
        print(f"iOS deployment package created at: {ios_package_dir}")
        return str(ios_package_dir)
    
    def create_android_deployment_package(self, model_dir: str, output_dir: str) -> str:
        """Create Android deployment package."""
        print("Creating Android deployment package...")
        
        android_package_dir = Path(output_dir) / "android_deployment"
        android_package_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model files
        model_files = list(Path(model_dir).glob("*.tflite"))
        if not model_files:
            raise ValueError("No TensorFlow Lite model files found in model directory")
        
        # Create assets directory
        assets_dir = android_package_dir / "app" / "src" / "main" / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        
        for model_file in model_files:
            shutil.copy2(model_file, assets_dir)
            print(f"Copied model: {model_file.name}")
        
        # Copy tokenizer files
        tokenizer_dir = Path(model_dir) / "tokenizer"
        if tokenizer_dir.exists():
            shutil.copytree(tokenizer_dir, assets_dir / "tokenizer", dirs_exist_ok=True)
        
        # Create Android integration code
        self.create_android_integration_code(android_package_dir)
        
        # Create Android project template
        self.create_android_project_template(android_package_dir)
        
        # Create documentation
        self.create_android_documentation(android_package_dir)
        
        print(f"Android deployment package created at: {android_package_dir}")
        return str(android_package_dir)
    
    def create_ios_integration_code(self, package_dir: Path):
        """Create iOS integration code."""
        swift_dir = package_dir / "Sources" / "BERTMobile"
        swift_dir.mkdir(parents=True, exist_ok=True)
        
        # Main BERT class
        bert_swift = '''import Foundation
import CoreML
import NaturalLanguage

@available(iOS 15.0, *)
public class BERTMobile {
    private var model: MLModel?
    private let tokenizer: BERTTokenizer
    private let modelName = "BERTMobile"
    
    public init() throws {
        // Load tokenizer
        tokenizer = try BERTTokenizer()
        
        // Load CoreML model
        try loadModel()
    }
    
    private func loadModel() throws {
        guard let modelURL = Bundle.module.url(forResource: modelName, withExtension: "mlmodel") else {
            throw BERTError.modelNotFound
        }
        
        let config = MLModelConfiguration()
        config.computeUnits = .all
        
        model = try MLModel(contentsOf: modelURL, configuration: config)
    }
    
    public func predict(text: String) throws -> BERTPrediction {
        guard let model = model else {
            throw BERTError.modelNotLoaded
        }
        
        // Tokenize input
        let tokens = try tokenizer.tokenize(text: text, maxLength: 128)
        
        // Create MLMultiArrays
        let inputIds = try MLMultiArray(shape: [1, 128], dataType: .int32)
        let attentionMask = try MLMultiArray(shape: [1, 128], dataType: .int32)
        
        for i in 0..<128 {
            inputIds[i] = NSNumber(value: i < tokens.inputIds.count ? tokens.inputIds[i] : 0)
            attentionMask[i] = NSNumber(value: i < tokens.attentionMask.count ? tokens.attentionMask[i] : 0)
        }
        
        // Create input dictionary
        let input: [String: Any] = [
            "input_ids": inputIds,
            "attention_mask": attentionMask
        ]
        
        // Run prediction
        let output = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: input))
        
        // Process output
        return try BERTPrediction(from: output)
    }
    
    public func classifyText(_ text: String) throws -> (label: String, confidence: Float) {
        let prediction = try predict(text: text)
        
        let labels = ["Negative", "Positive"] // Adjust based on your model
        let label = labels[prediction.predictedClass]
        
        return (label: label, confidence: prediction.confidence)
    }
}

public struct BERTTokens {
    public let inputIds: [Int32]
    public let attentionMask: [Int32]
}

public struct BERTPrediction {
    public let logits: [Float]
    public let predictedClass: Int
    public let confidence: Float
    
    init(from output: MLFeatureProvider) throws {
        guard let outputArray = output.featureValue(for: "output")?.multiArrayValue else {
            throw BERTError.invalidOutput
        }
        
        // Convert MLMultiArray to Float array
        logits = (0..<outputArray.count).map { Float(truncating: outputArray[$0]) }
        
        // Apply softmax and find predicted class
        let expLogits = logits.map { exp($0) }
        let sumExp = expLogits.reduce(0, +)
        let probabilities = expLogits.map { $0 / sumExp }
        
        let maxIndex = probabilities.enumerated().max(by: { $0.1 < $1.1 })?.0 ?? 0
        predictedClass = maxIndex
        confidence = probabilities[maxIndex]
    }
}

public enum BERTError: Error {
    case modelNotFound
    case modelNotLoaded
    case tokenizerError
    case invalidOutput
    case invalidInput
}'''
        
        with open(swift_dir / "BERTMobile.swift", 'w') as f:
            f.write(bert_swift)
        
        # Tokenizer class
        tokenizer_swift = '''import Foundation

public class BERTTokenizer {
    private let vocabulary: [String: Int]
    private let reverseVocabulary: [Int: String]
    private let specialTokens: [String: Int]
    
    public init() throws {
        // Load vocabulary
        guard let vocabURL = Bundle.module.url(forResource: "vocab", withExtension: "txt") else {
            throw BERTError.tokenizerError
        }
        
        let vocabContent = try String(contentsOf: vocabURL)
        let tokens = vocabContent.components(separatedBy: .newlines).filter { !$0.isEmpty }
        
        vocabulary = Dictionary(uniqueKeysWithValues: tokens.enumerated().map { ($1, $0) })
        reverseVocabulary = Dictionary(uniqueKeysWithValues: vocabulary.map { ($1, $0) })
        
        // Load special tokens
        specialTokens = [
            "[PAD]": 0,
            "[UNK]": 100,
            "[CLS]": 101,
            "[SEP]": 102,
            "[MASK]": 103
        ]
    }
    
    public func tokenize(text: String, maxLength: Int = 128) throws -> BERTTokens {
        // Simple tokenization - in production, use proper WordPiece tokenization
        let cleanText = text.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        let words = cleanText.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
        
        var inputIds: [Int32] = [Int32(specialTokens["[CLS]"] ?? 101)]
        var attentionMask: [Int32] = [1]
        
        for word in words {
            if inputIds.count >= maxLength - 1 { break }
            
            let tokenId = vocabulary[word] ?? specialTokens["[UNK]"] ?? 100
            inputIds.append(Int32(tokenId))
            attentionMask.append(1)
        }
        
        // Add SEP token
        if inputIds.count < maxLength {
            inputIds.append(Int32(specialTokens["[SEP]"] ?? 102))
            attentionMask.append(1)
        }
        
        // Pad to maxLength
        while inputIds.count < maxLength {
            inputIds.append(Int32(specialTokens["[PAD]"] ?? 0))
            attentionMask.append(0)
        }
        
        return BERTTokens(inputIds: inputIds, attentionMask: attentionMask)
    }
    
    public func decode(tokenIds: [Int32]) -> String {
        let tokens = tokenIds.compactMap { reverseVocabulary[Int($0)] }
            .filter { !["[PAD]", "[CLS]", "[SEP]"].contains($0) }
        
        return tokens.joined(separator: " ")
    }
}'''
        
        with open(swift_dir / "BERTTokenizer.swift", 'w') as f:
            f.write(tokenizer_swift)
        
        # Package.swift
        package_swift = '''// swift-tools-version:5.5
import PackageDescription

let package = Package(
    name: "BERTMobile",
    platforms: [
        .iOS(.v15)
    ],
    products: [
        .library(
            name: "BERTMobile",
            targets: ["BERTMobile"]
        ),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "BERTMobile",
            dependencies: [],
            resources: [
                .process("Resources")
            ]
        ),
        .testTarget(
            name: "BERTMobileTests",
            dependencies: ["BERTMobile"]
        ),
    ]
)'''
        
        with open(package_dir / "Package.swift", 'w') as f:
            f.write(package_swift)
    
    def create_android_integration_code(self, package_dir: Path):
        """Create Android integration code."""
        java_dir = package_dir / "app" / "src" / "main" / "java" / "com" / "bertmobile"
        java_dir.mkdir(parents=True, exist_ok=True)
        
        # Main BERT class in Kotlin
        bert_kotlin = '''package com.bertmobile

import android.content.Context
import android.content.res.AssetManager
import org.tensorflow.lite.Interpreter
import org.json.JSONObject
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.io.InputStream
import kotlin.math.exp

class BERTMobile(private val context: Context) {
    private var interpreter: Interpreter? = null
    private val tokenizer: BERTTokenizer by lazy { BERTTokenizer(context) }
    
    init {
        loadModel()
    }
    
    private fun loadModel() {
        try {
            val model = loadModelFile("bert_mobile.tflite")
            val options = Interpreter.Options()
            options.setNumThreads(4)
            options.setUseXNNPACK(true)
            
            // Try to use GPU delegate if available
            try {
                options.addDelegate(org.tensorflow.lite.gpu.GpuDelegate())
                println("Using GPU delegate")
            } catch (e: Exception) {
                println("GPU delegate not available, using CPU: ${e.message}")
            }
            
            // Try to use NNAPI delegate if available
            try {
                options.addDelegate(org.tensorflow.lite.nnapi.NnApiDelegate())
                println("Using NNAPI delegate")
            } catch (e: Exception) {
                println("NNAPI delegate not available: ${e.message}")
            }
            
            interpreter = Interpreter(model, options)
            println("BERT model loaded successfully")
            
        } catch (e: Exception) {
            throw RuntimeException("Failed to load BERT model", e)
        }
    }
    
    private fun loadModelFile(fileName: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(fileName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    fun predict(text: String): BERTPrediction {
        val interpreter = this.interpreter ?: throw IllegalStateException("Model not loaded")
        
        // Tokenize input
        val tokens = tokenizer.tokenize(text, maxLength = 128)
        
        // Prepare input arrays
        val inputIds = Array(1) { IntArray(128) }
        val attentionMask = Array(1) { IntArray(128) }
        
        for (i in 0 until 128) {
            inputIds[0][i] = if (i < tokens.inputIds.size) tokens.inputIds[i] else 0
            attentionMask[0][i] = if (i < tokens.attentionMask.size) tokens.attentionMask[i] else 0
        }
        
        // Prepare output array - adjust size based on your model
        val output = Array(1) { FloatArray(2) } // For binary classification
        
        // Run inference
        interpreter.run(arrayOf(inputIds, attentionMask), output)
        
        return BERTPrediction(output[0])
    }
    
    fun classifyText(text: String): Pair<String, Float> {
        val prediction = predict(text)
        val labels = arrayOf("Negative", "Positive") // Adjust based on your model
        val label = labels[prediction.predictedClass]
        return Pair(label, prediction.confidence)
    }
    
    fun close() {
        interpreter?.close()
        interpreter = null
    }
}

data class BERTTokens(
    val inputIds: IntArray,
    val attentionMask: IntArray
)

data class BERTPrediction(
    val logits: FloatArray
) {
    val predictedClass: Int
    val confidence: Float
    
    init {
        // Apply softmax to get probabilities
        val expLogits = logits.map { exp(it) }
        val sumExp = expLogits.sum()
        val probabilities = expLogits.map { it / sumExp }
        
        predictedClass = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
        confidence = probabilities[predictedClass]
    }
}'''
        
        with open(java_dir / "BERTMobile.kt", 'w') as f:
            f.write(bert_kotlin)
        
        # Tokenizer class
        tokenizer_kotlin = '''package com.bertmobile

import android.content.Context
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader

class BERTTokenizer(private val context: Context) {
    private val vocabulary: Map<String, Int>
    private val reverseVocabulary: Map<Int, String>
    private val specialTokens: Map<String, Int>
    
    init {
        // Load vocabulary
        vocabulary = loadVocabulary()
        reverseVocabulary = vocabulary.entries.associate { it.value to it.key }
        
        // Load special tokens
        specialTokens = mapOf(
            "[PAD]" to 0,
            "[UNK]" to 100,
            "[CLS]" to 101,
            "[SEP]" to 102,
            "[MASK]" to 103
        )
    }
    
    private fun loadVocabulary(): Map<String, Int> {
        return try {
            val inputStream = context.assets.open("tokenizer/vocab.json")
            val reader = BufferedReader(InputStreamReader(inputStream))
            val jsonString = reader.readText()
            reader.close()
            
            val jsonObject = JSONObject(jsonString)
            val vocab = mutableMapOf<String, Int>()
            
            val keys = jsonObject.keys()
            while (keys.hasNext()) {
                val key = keys.next()
                vocab[key] = jsonObject.getInt(key)
            }
            
            vocab
        } catch (e: Exception) {
            // Fallback to basic vocabulary if file not found
            println("Could not load vocabulary file, using fallback: ${e.message}")
            createFallbackVocabulary()
        }
    }
    
    private fun createFallbackVocabulary(): Map<String, Int> {
        val vocab = mutableMapOf<String, Int>()
        // Add special tokens
        specialTokens.forEach { (token, id) -> vocab[token] = id }
        
        // Add some common words (simplified for demo)
        val commonWords = listOf("the", "a", "an", "and", "or", "but", "in", "on", "at", "to", 
                               "for", "of", "with", "by", "is", "are", "was", "were", "be", 
                               "been", "being", "have", "has", "had", "do", "does", "did",
                               "will", "would", "could", "should", "this", "that", "these", "those")
        
        commonWords.forEachIndexed { index, word ->
            vocab[word] = 200 + index
        }
        
        return vocab
    }
    
    fun tokenize(text: String, maxLength: Int = 128): BERTTokens {
        // Simple tokenization - in production, use proper WordPiece tokenization
        val cleanText = text.lowercase().trim()
        val words = cleanText.split("\\s+".toRegex()).filter { it.isNotEmpty() }
        
        val inputIds = mutableListOf<Int>()
        val attentionMask = mutableListOf<Int>()
        
        // Add CLS token
        inputIds.add(specialTokens["[CLS]"] ?: 101)
        attentionMask.add(1)
        
        // Add word tokens
        for (word in words) {
            if (inputIds.size >= maxLength - 1) break
            
            val tokenId = vocabulary[word] ?: specialTokens["[UNK]"] ?: 100
            inputIds.add(tokenId)
            attentionMask.add(1)
        }
        
        // Add SEP token
        if (inputIds.size < maxLength) {
            inputIds.add(specialTokens["[SEP]"] ?: 102)
            attentionMask.add(1)
        }
        
        // Pad to maxLength
        while (inputIds.size < maxLength) {
            inputIds.add(specialTokens["[PAD]"] ?: 0)
            attentionMask.add(0)
        }
        
        return BERTTokens(
            inputIds = inputIds.toIntArray(),
            attentionMask = attentionMask.toIntArray()
        )
    }
    
    fun decode(tokenIds: IntArray): String {
        val tokens = tokenIds.mapNotNull { tokenId ->
            when (tokenId) {
                specialTokens["[PAD]"] -> null // Skip padding tokens
                specialTokens["[CLS]"] -> null // Skip CLS token  
                specialTokens["[SEP]"] -> null // Skip SEP token
                else -> reverseVocabulary[tokenId] ?: "[UNK]"
            }
        }
        
        return tokens.joinToString(" ")
    }
}'''
        
        with open(java_dir / "BERTTokenizer.kt", 'w') as f:
            f.write(tokenizer_kotlin)
    
    def create_ios_project_template(self, package_dir: Path):
        """Create iOS project template."""
        examples_dir = package_dir / "Examples"
        examples_dir.mkdir(exist_ok=True)
        
        # Example SwiftUI app
        example_app = '''import SwiftUI
import BERTMobile

@main
struct BERTMobileApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

struct ContentView: View {
    @State private var inputText = ""
    @State private var result = ""
    @State private var isProcessing = false
    
    private var bertModel: BERTMobile? = {
        do {
            return try BERTMobile()
        } catch {
            print("Failed to load BERT model: \\(error)")
            return nil
        }
    }()
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                Text("BERT Mobile Demo")
                    .font(.title)
                    .padding()
                
                VStack(alignment: .leading, spacing: 8) {
                    Text("Enter text to classify:")
                        .font(.headline)
                    
                    TextEditor(text: $inputText)
                        .frame(height: 100)
                        .border(Color.gray, width: 1)
                        .padding(.horizontal)
                }
                
                Button("Classify Text") {
                    classifyText()
                }
                .disabled(inputText.isEmpty || isProcessing)
                .padding()
                .background(inputText.isEmpty || isProcessing ? Color.gray : Color.blue)
                .foregroundColor(.white)
                .cornerRadius(8)
                
                if isProcessing {
                    ProgressView("Processing...")
                        .padding()
                } else if !result.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Result:")
                            .font(.headline)
                        
                        Text(result)
                            .padding()
                            .background(Color.gray.opacity(0.1))
                            .cornerRadius(8)
                    }
                    .padding(.horizontal)
                }
                
                Spacer()
            }
            .navigationTitle("BERT Mobile")
        }
    }
    
    private func classifyText() {
        guard let model = bertModel else {
            result = "Error: Model not loaded"
            return
        }
        
        isProcessing = true
        
        DispatchQueue.global().async {
            do {
                let startTime = Date()
                let (label, confidence) = try model.classifyText(inputText)
                let endTime = Date()
                let duration = endTime.timeIntervalSince(startTime)
                
                DispatchQueue.main.async {
                    result = """
                    Classification: \\(label)
                    Confidence: \\(String(format: "%.2f%%", confidence * 100))
                    Processing time: \\(String(format: "%.0f", duration * 1000))ms
                    """
                    isProcessing = false
                }
            } catch {
                DispatchQueue.main.async {
                    result = "Error: \\(error.localizedDescription)"
                    isProcessing = false
                }
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}'''
        
        with open(examples_dir / "ContentView.swift", 'w') as f:
            f.write(example_app)
    
    def create_android_project_template(self, package_dir: Path):
        """Create Android project template."""
        examples_dir = package_dir / "examples"
        examples_dir.mkdir(exist_ok=True)
        
        # Example MainActivity
        main_activity = '''package com.bertmobile.example

import android.os.Bundle
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.bertmobile.BERTMobile
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {
    
    private lateinit var bertMobile: BERTMobile
    private lateinit var inputEditText: EditText
    private lateinit var classifyButton: Button
    private lateinit var resultTextView: TextView
    private lateinit var progressBar: ProgressBar
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        initViews()
        initBERTModel()
        setupClickListeners()
    }
    
    private fun initViews() {
        inputEditText = findViewById(R.id.inputEditText)
        classifyButton = findViewById(R.id.classifyButton)
        resultTextView = findViewById(R.id.resultTextView)
        progressBar = findViewById(R.id.progressBar)
        
        // Set some sample text
        inputEditText.setText("This is a great product! I highly recommend it.")
    }
    
    private fun initBERTModel() {
        try {
            bertMobile = BERTMobile(this)
            Toast.makeText(this, "BERT model loaded successfully", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Toast.makeText(this, "Failed to load BERT model: ${e.message}", Toast.LENGTH_LONG).show()
            e.printStackTrace()
        }
    }
    
    private fun setupClickListeners() {
        classifyButton.setOnClickListener {
            val inputText = inputEditText.text.toString().trim()
            if (inputText.isNotEmpty()) {
                classifyText(inputText)
            } else {
                Toast.makeText(this, "Please enter some text", Toast.LENGTH_SHORT).show()
            }
        }
    }
    
    private fun classifyText(text: String) {
        classifyButton.isEnabled = false
        progressBar.visibility = ProgressBar.VISIBLE
        resultTextView.text = "Processing..."
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.Default) {
                    val startTime = System.currentTimeMillis()
                    val (label, confidence) = bertMobile.classifyText(text)
                    val endTime = System.currentTimeMillis()
                    val duration = endTime - startTime
                    
                    """
                    Classification: $label
                    Confidence: ${String.format("%.2f%%", confidence * 100)}
                    Processing time: ${duration}ms
                    
                    Input text:
                    "$text"
                    """.trimIndent()
                }
                
                resultTextView.text = result
                
            } catch (e: Exception) {
                resultTextView.text = "Error: ${e.message}"
                e.printStackTrace()
            } finally {
                classifyButton.isEnabled = true
                progressBar.visibility = ProgressBar.GONE
            }
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        if (::bertMobile.isInitialized) {
            bertMobile.close()
        }
    }
}'''
        
        with open(examples_dir / "MainActivity.kt", 'w') as f:
            f.write(main_activity)
        
        # Layout file
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
            android:text="BERT Mobile Demo"
            android:textSize="24sp"
            android:textStyle="bold"
            android:layout_marginBottom="24dp"
            android:layout_gravity="center_horizontal" />

        <TextView
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:text="Enter text to classify:"
            android:textSize="16sp"
            android:textStyle="bold"
            android:layout_marginBottom="8dp" />

        <EditText
            android:id="@+id/inputEditText"
            android:layout_width="match_parent"
            android:layout_height="120dp"
            android:gravity="top|start"
            android:hint="Enter text here..."
            android:inputType="textMultiLine"
            android:background="@android:drawable/edit_text"
            android:padding="12dp"
            android:layout_marginBottom="16dp" />

        <Button
            android:id="@+id/classifyButton"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:text="Classify Text"
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
            android:text="Result:"
            android:textSize="16sp"
            android:textStyle="bold"
            android:layout_marginBottom="8dp" />

        <TextView
            android:id="@+id/resultTextView"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:text="Results will appear here..."
            android:background="#f9f9f9"
            android:padding="12dp"
            android:textIsSelectable="true"
            android:fontFamily="monospace"
            android:textSize="14sp"
            android:minHeight="120dp"
            android:gravity="top|start" />

    </LinearLayout>

</ScrollView>'''
        
        with open(examples_dir / "activity_main.xml", 'w') as f:
            f.write(layout_xml)
    
    def create_ios_documentation(self, package_dir: Path):
        """Create iOS documentation."""
        docs_dir = package_dir / "Documentation"
        docs_dir.mkdir(exist_ok=True)
        
        readme = '''# BERT Mobile iOS Integration

## Overview

This package provides a complete BERT model integration for iOS applications using CoreML.

## Requirements

- iOS 15.0+
- Xcode 13.0+
- Swift 5.5+

## Installation

### Swift Package Manager

Add to your Package.swift:

```swift
dependencies: [
    .package(url: "path/to/BERTMobile", from: "1.0.0")
]
```

### Manual Installation

1. Copy the `BERTMobile.mlmodel` file to your Xcode project
2. Add the Swift source files to your project
3. Import the BERTMobile module

## Quick Start

```swift
import BERTMobile

// Initialize BERT model
do {
    let bert = try BERTMobile()
    
    // Classify text
    let (label, confidence) = try bert.classifyText("This is a great product!")
    print("Classification: \\(label) (\\(confidence * 100)% confidence)")
    
} catch {
    print("Error: \\(error)")
}
```

## Advanced Usage

### Custom Prediction

```swift
let bert = try BERTMobile()
let prediction = try bert.predict(text: "Your text here")

print("Logits: \\(prediction.logits)")
print("Predicted class: \\(prediction.predictedClass)")
print("Confidence: \\(prediction.confidence)")
```

### Batch Processing

```swift
let texts = ["Text 1", "Text 2", "Text 3"]
let results = try texts.map { try bert.classifyText($0) }
```

## Performance

- Average inference time: ~50-150ms on modern devices
- Memory usage: ~50-80MB
- Model size: ~25-50MB

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure the .mlmodel file is added to your bundle
2. **Slow performance**: Check that Neural Engine is being used
3. **Memory issues**: Monitor memory usage and consider batching

### Performance Tips

1. Reuse the BERTMobile instance
2. Process text on background queues
3. Use shorter text when possible
4. Enable Neural Engine optimization

## API Reference

### BERTMobile

Main class for BERT model inference.

#### Methods

- `init() throws`: Initialize the model
- `predict(text: String) throws -> BERTPrediction`: Get raw prediction
- `classifyText(_ text: String) throws -> (label: String, confidence: Float)`: Classify text

### BERTPrediction

Result structure containing prediction details.

#### Properties

- `logits: [Float]`: Raw model outputs
- `predictedClass: Int`: Index of predicted class
- `confidence: Float`: Confidence score (0-1)

### BERTError

Error types that can occur during model operations.

#### Cases

- `modelNotFound`: Model file not found in bundle
- `modelNotLoaded`: Model failed to load
- `tokenizerError`: Tokenization failed
- `invalidOutput`: Model output format error
- `invalidInput`: Input format error
'''
        
        with open(docs_dir / "README.md", 'w') as f:
            f.write(readme)
    
    def create_android_documentation(self, package_dir: Path):
        """Create Android documentation."""
        docs_dir = package_dir / "Documentation"
        docs_dir.mkdir(exist_ok=True)
        
        readme = '''# BERT Mobile Android Integration

## Overview

This package provides a complete BERT model integration for Android applications using TensorFlow Lite.

## Requirements

- Android API 24+ (Android 7.0)
- Kotlin 1.7+
- TensorFlow Lite 2.13+

## Installation

Add to your app's `build.gradle`:

```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3'
}
```

## Setup

1. Copy `bert_mobile.tflite` to `app/src/main/assets/`
2. Copy tokenizer files to `app/src/main/assets/tokenizer/`
3. Add the BERTMobile.kt and BERTTokenizer.kt files to your project

## Quick Start

```kotlin
class MainActivity : AppCompatActivity() {
    private lateinit var bertMobile: BERTMobile
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Initialize BERT model
        bertMobile = BERTMobile(this)
        
        // Classify text
        lifecycleScope.launch {
            val (label, confidence) = bertMobile.classifyText("This is a great product!")
            println("Classification: $label (${confidence * 100}% confidence)")
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        bertMobile.close() // Clean up resources
    }
}
```

## Advanced Usage

### Custom Prediction

```kotlin
val prediction = bertMobile.predict("Your text here")
println("Logits: ${prediction.logits.contentToString()}")
println("Predicted class: ${prediction.predictedClass}")
println("Confidence: ${prediction.confidence}")
```

### Batch Processing

```kotlin
val texts = listOf("Text 1", "Text 2", "Text 3")
val results = texts.map { bertMobile.classifyText(it) }
```

### Background Processing

```kotlin
lifecycleScope.launch(Dispatchers.Default) {
    val result = bertMobile.classifyText(longText)
    
    withContext(Dispatchers.Main) {
        // Update UI with result
        updateUI(result)
    }
}
```

## Performance

- Average inference time: ~100-300ms (device dependent)
- Memory usage: ~100-150MB
- Model size: ~25-50MB

## Optimization

### GPU Acceleration

The library automatically tries to use:
1. GPU Delegate (if available)
2. NNAPI Delegate (if supported)
3. CPU fallback

### Performance Tips

1. Keep the BERTMobile instance alive
2. Use coroutines for background processing
3. Enable hardware acceleration
4. Batch similar requests

## Troubleshooting

### Common Issues

1. **Model not loading**:
   - Check that .tflite file is in assets folder
   - Verify file permissions

2. **Slow performance**:
   - Ensure GPU delegate is working
   - Check device compatibility

3. **Memory issues**:
   - Call `close()` when done
   - Monitor memory usage

### Performance Monitoring

```kotlin
val startTime = System.currentTimeMillis()
val result = bertMobile.predict(text)
val duration = System.currentTimeMillis() - startTime
Log.d("BERT", "Inference took ${duration}ms")
```

## API Reference

### BERTMobile

#### Constructor
- `BERTMobile(context: Context)`: Initialize with Android context

#### Methods
- `predict(text: String): BERTPrediction`: Get raw prediction
- `classifyText(text: String): Pair<String, Float>`: Classify text
- `close()`: Release resources

### BERTPrediction

#### Properties
- `logits: FloatArray`: Raw model outputs
- `predictedClass: Int`: Index of predicted class  
- `confidence: Float`: Confidence score (0-1)

### BERTTokenizer

#### Methods
- `tokenize(text: String, maxLength: Int = 128): BERTTokens`: Tokenize text
- `decode(tokenIds: IntArray): String`: Decode token IDs back to text

## Model Information

- Input: Text sequences up to 128 tokens
- Output: Classification logits
- Quantization: INT8 for mobile optimization
- Hardware: Optimized for ARM64 and GPU acceleration

## Support

For issues and questions:
1. Check the troubleshooting section
2. Verify model and tokenizer files
3. Test with sample inputs
4. Monitor device performance
'''
        
        with open(docs_dir / "README.md", 'w') as f:
            f.write(readme)

def main():
    parser = argparse.ArgumentParser(description='Deploy BERT models to mobile platforms')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing converted mobile models')
    parser.add_argument('--platform', type=str, choices=['ios', 'android', 'both'], 
                       required=True, help='Target deployment platform')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for deployment package')
    parser.add_argument('--config', type=str, default='config/mobile_config.yaml',
                       help='Path to mobile configuration')
    parser.add_argument('--create_archive', action='store_true',
                       help='Create compressed archive of deployment package')
    
    args = parser.parse_args()
    
    try:
        print("Preparing BERT mobile deployment...")
        
        # Initialize deployment manager
        deployment = BERTDeploymentManager(args.config)
        
        platforms = ['ios', 'android'] if args.platform == 'both' else [args.platform]
        created_packages = []
        
        for platform in platforms:
            model_platform_dir = os.path.join(args.model_dir, platform)
            
            if not os.path.exists(model_platform_dir):
                print(f"Warning: No {platform} models found in {model_platform_dir}")
                continue
            
            if platform == 'ios':
                package_path = deployment.create_ios_deployment_package(
                    model_platform_dir, args.output_dir
                )
            elif platform == 'android':
                package_path = deployment.create_android_deployment_package(
                    model_platform_dir, args.output_dir
                )
            
            created_packages.append((platform, package_path))
        
        # Create archives if requested
        if args.create_archive:
            for platform, package_path in created_packages:
                archive_path = f"{package_path}.zip"
                shutil.make_archive(package_path, 'zip', package_path)
                print(f"Created archive: {archive_path}")
        
        print("\nDeployment packages created successfully!")
        for platform, package_path in created_packages:
            print(f"  {platform.upper()}: {package_path}")
        
        print("\nNext steps:")
        print("1. Review the generated code and documentation")
        print("2. Test the example applications")
        print("3. Integrate into your mobile app")
        print("4. Customize the UI and functionality as needed")
        
    except Exception as e:
        print(f"Error during deployment preparation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
