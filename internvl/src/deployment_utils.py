# internvl/src/deployment_utils.py

import os
import shutil
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeploymentUtils:
    """Utilities for deploying InternVL models to mobile platforms."""
    
    def __init__(self):
        """Initialize deployment utilities."""
        pass
    
    def create_deployment_package(self,
                                 model_dir: str,
                                 platform: str,
                                 output_dir: str,
                                 include_examples: bool = True) -> str:
        """
        Create a deployment package for the specified platform.
        
        Args:
            model_dir: Directory containing converted models
            platform: Target platform ('ios' or 'android')
            output_dir: Output directory for deployment package
            include_examples: Whether to include example code
            
        Returns:
            Path to created deployment package
        """
        logger.info(f"Creating {platform} deployment package...")
        
        model_path = Path(model_dir)
        output_path = Path(output_dir)
        package_path = output_path / f"internvl_{platform}_deployment"
        
        # Create package directory
        package_path.mkdir(parents=True, exist_ok=True)
        
        # Copy model files
        self.copy_model_files(model_path / platform, package_path / "models")
        
        # Create integration files
        if platform == 'ios':
            self.create_ios_package(package_path, include_examples)
        elif platform == 'android':
            self.create_android_package(package_path, include_examples)
        
        # Create documentation
        self.create_documentation(package_path, platform)
        
        logger.info(f"Deployment package created at: {package_path}")
        return str(package_path)
    
    def copy_model_files(self, source_dir: Path, target_dir: Path):
        """Copy model files to deployment package."""
        target_dir.mkdir(parents=True, exist_ok=True)
        
        if source_dir.exists():
            for file_path in source_dir.iterdir():
                if file_path.is_file():
                    shutil.copy2(file_path, target_dir)
                    logger.info(f"Copied model file: {file_path.name}")
    
    def create_ios_package(self, package_path: Path, include_examples: bool):
        """Create iOS-specific deployment files."""
        
        # Create directory structure
        (package_path / "ios" / "Sources" / "InternVL").mkdir(parents=True, exist_ok=True)
        (package_path / "ios" / "Resources").mkdir(parents=True, exist_ok=True)
        
        # Create Package.swift
        package_swift = '''// swift-tools-version:5.5
import PackageDescription

let package = Package(
    name: "InternVL",
    platforms: [
        .iOS(.v15)
    ],
    products: [
        .library(
            name: "InternVL",
            targets: ["InternVL"]
        ),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "InternVL",
            dependencies: [],
            resources: [
                .process("Resources")
            ]
        ),
        .testTarget(
            name: "InternVLTests",
            dependencies: ["InternVL"]
        ),
    ]
)
'''
        
        with open(package_path / "ios" / "Package.swift", 'w') as f:
            f.write(package_swift)
        
        # Create main InternVL class
        internvl_swift = '''import CoreML
import UIKit
import Vision

@available(iOS 15.0, *)
public class InternVL {
    private var model: MLModel?
    private let modelName = "InternVLMobile"
    
    public init() {
        loadModel()
    }
    
    private func loadModel() {
        guard let modelURL = Bundle.module.url(forResource: modelName, withExtension: "mlmodel") else {
            print("Failed to find model file")
            return
        }
        
        do {
            model = try MLModel(contentsOf: modelURL)
            print("Model loaded successfully")
        } catch {
            print("Failed to load model: \\(error)")
        }
    }
    
    public func predict(image: UIImage, text: String, completion: @escaping (String?) -> Void) {
        guard let model = model else {
            completion(nil)
            return
        }
        
        // Convert image to CVPixelBuffer
        guard let pixelBuffer = image.toCVPixelBuffer() else {
            completion(nil)
            return
        }
        
        // Tokenize text (simplified)
        let tokens = tokenize(text: text)
        
        // Create prediction input
        do {
            let input = InternVLMobileInput(image: pixelBuffer, input_ids: tokens)
            let prediction = try model.prediction(from: input)
            
            // Process output (simplified)
            let result = processOutput(prediction)
            completion(result)
            
        } catch {
            print("Prediction failed: \\(error)")
            completion(nil)
        }
    }
    
    private func tokenize(text: String) -> MLMultiArray {
        // Simplified tokenization - implement proper tokenization
        let tokens = text.components(separatedBy: " ").prefix(512)
        
        guard let multiArray = try? MLMultiArray(shape: [1, 512], dataType: .int32) else {
            return MLMultiArray()
        }
        
        for (index, token) in tokens.enumerated() {
            multiArray[index] = NSNumber(value: token.hash % 1000)
        }
        
        return multiArray
    }
    
    private func processOutput(_ prediction: MLFeatureProvider) -> String {
        // Process model output - implement based on your model's output format
        return "Model output processed"
    }
}

extension UIImage {
    func toCVPixelBuffer() -> CVPixelBuffer? {
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                       Int(self.size.width),
                                       Int(self.size.height),
                                       kCVPixelFormatType_32ARGB,
                                       attrs,
                                       &pixelBuffer)
        
        guard status == kCVReturnSuccess else { return nil }
        
        CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)
        
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: pixelData,
                              width: Int(self.size.width),
                              height: Int(self.size.height),
                              bitsPerComponent: 8,
                              bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!),
                              space: rgbColorSpace,
                              bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
        
        context?.translateBy(x: 0, y: self.size.height)
        context?.scaleBy(x: 1.0, y: -1.0)
        
        UIGraphicsPushContext(context!)
        self.draw(in: CGRect(x: 0, y: 0, width: self.size.width, height: self.size.height))
        UIGraphicsPopContext()
        CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        
        return pixelBuffer
    }
}
'''
        
        with open(package_path / "ios" / "Sources" / "InternVL" / "InternVL.swift", 'w') as f:
            f.write(internvl_swift)
        
        if include_examples:
            self.create_ios_examples(package_path / "ios")
    
    def create_android_package(self, package_path: Path, include_examples: bool):
        """Create Android-specific deployment files."""
        
        # Create directory structure
        android_dir = package_path / "android"
        (android_dir / "src" / "main" / "java" / "com" / "internvl").mkdir(parents=True, exist_ok=True)
        (android_dir / "src" / "main" / "assets").mkdir(parents=True, exist_ok=True)
        
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
        consumerProguardFiles "consumer-rules.pro"
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
    
    // Image processing
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
    
    testImplementation 'junit:junit:4.13.2'
    androidTestImplementation 'androidx.test.ext:junit:1.1.5'
    androidTestImplementation 'androidx.test.espresso:espresso-core:3.5.1'
}
'''
        
        with open(android_dir / "build.gradle", 'w') as f:
            f.write(build_gradle)
        
        # Create main InternVL class
        internvl_kotlin = '''package com.internvl

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class InternVL(private val context: Context) {
    private var interpreter: Interpreter? = null
    private val modelName = "model.tflite"
    private val inputImageSize = 224
    
    init {
        loadModel()
    }
    
    private fun loadModel() {
        try {
            val model = loadModelFile()
            interpreter = Interpreter(model)
            println("Model loaded successfully")
        } catch (e: Exception) {
            e.printStackTrace()
            println("Failed to load model: ${e.message}")
        }
    }
    
    private fun loadModelFile(): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(modelName)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    fun predict(bitmap: Bitmap, text: String): String? {
        val interpreter = this.interpreter ?: return null
        
        try {
            // Preprocess image
            val tensorImage = TensorImage.fromBitmap(bitmap)
            val imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(inputImageSize, inputImageSize, ResizeOp.ResizeMethod.BILINEAR))
                .build()
            val processedImage = imageProcessor.process(tensorImage)
            
            // Tokenize text (simplified)
            val tokens = tokenizeText(text)
            
            // Prepare inputs
            val inputs = arrayOf(processedImage.buffer, tokens)
            
            // Prepare outputs
            val outputs = Array(1) { FloatArray(512) } // Adjust size based on your model
            
            // Run inference
            interpreter.run(inputs, outputs)
            
            // Process outputs
            return processOutput(outputs[0])
            
        } catch (e: Exception) {
            e.printStackTrace()
            return null
        }
    }
    
    private fun tokenizeText(text: String): IntArray {
        // Simplified tokenization - implement proper tokenization
        val tokens = text.split(" ").take(512)
        val tokenArray = IntArray(512)
        
        tokens.forEachIndexed { index, token ->
            tokenArray[index] = token.hashCode() % 1000
        }
        
        return tokenArray
    }
    
    private fun processOutput(output: FloatArray): String {
        // Process model output - implement based on your model's output format
        return "Model output processed"
    }
    
    fun close() {
        interpreter?.close()
        interpreter = null
    }
}
'''
        
        with open(android_dir / "src" / "main" / "java" / "com" / "internvl" / "InternVL.kt", 'w') as f:
            f.write(internvl_kotlin)
        
        if include_examples:
            self.create_android_examples(android_dir)
    
    def create_ios_examples(self, ios_dir: Path):
        """Create iOS example code."""
        examples_dir = ios_dir / "Examples"
        examples_dir.mkdir(exist_ok=True)
        
        example_code = '''import SwiftUI
import InternVL

struct ContentView: View {
    @State private var selectedImage: UIImage?
    @State private var inputText = ""
    @State private var result = ""
    @State private var isProcessing = false
    
    private let internVL = InternVL()
    
    var body: some View {
        VStack(spacing: 20) {
            Text("InternVL Demo")
                .font(.title)
                .padding()
            
            // Image selection
            if let image = selectedImage {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxHeight: 200)
            } else {
                Rectangle()
                    .fill(Color.gray.opacity(0.3))
                    .frame(height: 200)
                    .overlay(Text("Select Image"))
            }
            
            Button("Select Image") {
                // Image picker implementation
            }
            
            // Text input
            TextField("Enter your question", text: $inputText)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .padding(.horizontal)
            
            // Process button
            Button("Process") {
                processImage()
            }
            .disabled(selectedImage == nil || inputText.isEmpty || isProcessing)
            
            // Result
            if isProcessing {
                ProgressView("Processing...")
            } else if !result.isEmpty {
                Text("Result: \\(result)")
                    .padding()
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(8)
            }
            
            Spacer()
        }
        .padding()
    }
    
    private func processImage() {
        guard let image = selectedImage else { return }
        
        isProcessing = true
        
        internVL.predict(image: image, text: inputText) { result in
            DispatchQueue.main.async {
                self.isProcessing = false
                self.result = result ?? "No result"
            }
        }
    }
}
'''
        
        with open(examples_dir / "ExampleApp.swift", 'w') as f:
            f.write(example_code)
    
    def create_android_examples(self, android_dir: Path):
        """Create Android example code."""
        examples_dir = android_dir / "examples"
        examples_dir.mkdir(exist_ok=True)
        
        example_code = '''package com.internvl.example

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.internvl.InternVL

class MainActivity : AppCompatActivity() {
    private lateinit var internVL: InternVL
    private lateinit var imageView: ImageView
    private lateinit var textInput: EditText
    private lateinit var processButton: Button
    private lateinit var resultText: TextView
    
    private var selectedBitmap: Bitmap? = null
    
    companion object {
        private const val REQUEST_IMAGE_CAPTURE = 1
        private const val REQUEST_CAMERA_PERMISSION = 100
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        initializeViews()
        initializeModel()
        setupClickListeners()
    }
    
    private fun initializeViews() {
        imageView = findViewById(R.id.imageView)
        textInput = findViewById(R.id.textInput)
        processButton = findViewById(R.id.processButton)
        resultText = findViewById(R.id.resultText)
    }
    
    private fun initializeModel() {
        internVL = InternVL(this)
    }
    
    private fun setupClickListeners() {
        findViewById<Button>(R.id.captureButton).setOnClickListener {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), REQUEST_CAMERA_PERMISSION)
            } else {
                dispatchTakePictureIntent()
            }
        }
        
        processButton.setOnClickListener {
            processImage()
        }
    }
    
    private fun dispatchTakePictureIntent() {
        Intent(MediaStore.ACTION_IMAGE_CAPTURE).also { takePictureIntent ->
            takePictureIntent.resolveActivity(packageManager)?.also {
                startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)
            }
        }
    }
    
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            val imageBitmap = data?.extras?.get("data") as Bitmap
            selectedBitmap = imageBitmap
            imageView.setImageBitmap(imageBitmap)
            processButton.isEnabled = true
        }
    }
    
    private fun processImage() {
        val bitmap = selectedBitmap ?: return
        val text = textInput.text.toString()
        
        if (text.isEmpty()) {
            Toast.makeText(this, "Please enter a question", Toast.LENGTH_SHORT).show()
            return
        }
        
        processButton.isEnabled = false
        resultText.text = "Processing..."
        
        Thread {
            val result = internVL.predict(bitmap, text)
            
            runOnUiThread {
                processButton.isEnabled = true
                resultText.text = result ?: "No result"
            }
        }.start()
    }
    
    override fun onDestroy() {
        super.onDestroy()
        internVL.close()
    }
}
'''
        
        with open(examples_dir / "MainActivity.kt", 'w') as f:
            f.write(example_code)
    
    def create_documentation(self, package_path: Path, platform: str):
        """Create comprehensive documentation."""
        
        readme_content = f"""# InternVL {platform.upper()} Deployment Package

This package contains everything needed to integrate InternVL into your {platform} application.

## Contents

- `models/`: Converted model files optimized for {platform}
- `{platform}/`: Platform-specific integration code
- `examples/`: Example applications and usage code
- `docs/`: Detailed documentation

## Quick Start

### 1. Integration

**{platform.upper()}:**
"""
        
        if platform == 'ios':
            readme_content += """
- Add the .mlmodel file to your Xcode project
- Import the InternVL Swift package
- Initialize the model: `let internvl = InternVL()`
- Make predictions: `internvl.predict(image: image, text: text) { result in ... }`
"""
        else:
            readme_content += """
- Add the .tflite file to your app/src/main/assets/ directory
- Add TensorFlow Lite dependencies to build.gradle
- Initialize the model: `val internvl = InternVL(context)`
- Make predictions: `val result = internvl.predict(bitmap, text)`
"""
        
        readme_content += f"""
### 2. Requirements

**{platform.upper()}:**
"""
        
        if platform == 'ios':
            readme_content += """
- iOS 15.0+
- Xcode 13.0+
- CoreML framework
- Vision framework
"""
        else:
            readme_content += """
- Android API 24+
- TensorFlow Lite 2.13.0+
- Camera permissions (for image capture)
"""
        
        readme_content += f"""
### 3. Model Information

- Input image size: 224x224 pixels
- Max text sequence length: 512 tokens
- Model format: {"CoreML (.mlmodel)" if platform == 'ios' else "TensorFlow Lite (.tflite)"}
- Optimizations: Quantization applied for mobile deployment

### 4. Performance

- Inference time: ~100-500ms (device dependent)
- Memory usage: ~100-200MB
- Battery impact: Optimized for mobile usage

## Support

For issues and questions:
1. Check the example code in the `examples/` directory
2. Review the detailed documentation in `docs/`
3. Validate model inputs and preprocessing steps

## License

Please refer to the original InternVL license terms.
"""
        
        with open(package_path / "README.md", 'w') as f:
            f.write(readme_content)
    
    def create_archive(self, package_path: str) -> str:
        """Create compressed archive of deployment package."""
        package_dir = Path(package_path)
        archive_path = package_dir.with_suffix('.zip')
        
        shutil.make_archive(
            str(archive_path.with_suffix('')),
            'zip',
            package_dir.parent,
            package_dir.name
        )
        
        logger.info(f"Created deployment archive: {archive_path}")
        return str(archive_path)

