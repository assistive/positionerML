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
            raise ValueError("No CoreML model files found")
        
        for model_file in model_files:
            shutil.copy2(model_file, ios_package_dir)
        
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
        
        return str(ios_package_dir)
    
    def create_android_deployment_package(self, model_dir: str, output_dir: str) -> str:
        """Create Android deployment package."""
        print("Creating Android deployment package...")
        
        android_package_dir = Path(output_dir) / "android_deployment"
        android_package_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model files
        model_files = list(Path(model_dir).glob("*.tflite"))
        if not model_files:
            raise ValueError("No TensorFlow Lite model files found")
        
        # Create assets directory
        assets_dir = android_package_dir / "app" / "src" / "main" / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        
        for model_file in model_files:
            shutil.copy2(model_file, assets_dir)
        
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
        
        // Find predicted class and confidence
        let maxIndex = logits.enumerated().max(by: { $0.1 < $1.1 })?.0 ?? 0
        predictedClass = maxIndex
        confidence = logits[maxIndex]
    }
}

public enum BERTError: Error {
    case modelNotFound
    case modelNotLoaded
    case tokenizerError
    case invalidOutput
    case invalidInput
}
'''
        
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
}
'''
        
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
)
'''
        
        with open(package_dir / "Package.swift", 'w') as f:
            f.write(package_swift)
    
    def create_android_integration_code(self, package_dir: Path):
        """Create Android integration code."""
        java_dir = package_dir / "app" / "src" / "main" / "java" / "com" / "bertmobile"
        java_dir.mkdir(parents=True, exist_ok=True)
        
        # Main BERT class
        bert_kotlin = '''package com.bertmobile

import android.content.Context
import android.content.res.AssetManager
import org.tensorflow.lite.Interpreter
import org.json.JSONObject
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.io.InputStream

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
            
            interpreter = Interpreter(model, options)
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
        
        // Prepare output array
        val output = Array(1) { FloatArray(2) } // Adjust size based on your model
        
        // Run inference
        interpreter.run(arrayOf(inputIds, attentionMask), output)
        
        return BERTPrediction(output[0])
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
    val predictedClass: Int = logits.indices.maxByOrNull { logits[it] } ?: 0
    val confidence: Float = logits.maxOrNull() ?: 0f
}
'''
        
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
        
        return vocab
    }
    
    fun tokenize(text: String, maxLength: Int = 128): BERTTokens {
        // Simple tokenization - in production, use proper WordPiece tokenization
        val cleanText = text.lowercase().trim()
        val words = cleanText.split("\\\\s+".toRegex()).filter { it.isNotEmpty() }
        
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
        
        // Pad to
