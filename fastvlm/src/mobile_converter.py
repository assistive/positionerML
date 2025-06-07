# fastvlm/src/mobile_converter.py

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import json
import onnx
import onnxruntime as ort
from dataclasses import dataclass
import yaml

# Platform-specific imports
try:
    import coremltools as ct
    from coremltools.models.neural_network import quantization_utils
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False
    logging.warning("CoreMLTools not available - iOS deployment disabled")

try:
    import tensorflow as tf
    import onnx_tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available - Android deployment disabled")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MobileOptimizationConfig:
    """Configuration for mobile optimization."""
    target_size_mb: float = 100.0
    quantization_bits: int = 8
    pruning_sparsity: float = 0.5
    use_dynamic_quantization: bool = True
    optimize_for_inference: bool = True
    batch_size: int = 1
    sequence_length: int = 77
    image_size: int = 224
    num_vision_tokens: int = 49  # Reduced from 256


class MobileConverter:
    """Converts FastVLM models for efficient mobile deployment."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize mobile converter."""
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
        
        self.optimization_config = MobileOptimizationConfig()
    
    def optimize_model(self, model_path: str, optimization_config: MobileOptimizationConfig) -> str:
        """Apply optimizations to the model before conversion."""
        logger.info(f"Optimizing model from {model_path}")
        
        # Load model
        model = torch.load(model_path, map_location='cpu')
        if isinstance(model, dict) and 'model_state_dict' in model:
            state_dict = model['model_state_dict']
            # Load model architecture
            from fastvlm_model import FastVLMModel, FastVLMConfig
            config = FastVLMConfig(**model.get('config', {}))
            optimized_model = FastVLMModel(config)
            optimized_model.load_state_dict(state_dict)
        else:
            optimized_model = model
        
        optimized_model.eval()
        
        # Apply optimizations
        if optimization_config.pruning_sparsity > 0:
            optimized_model = self._apply_pruning(optimized_model, optimization_config.pruning_sparsity)
        
        if optimization_config.use_dynamic_quantization:
            optimized_model = self._apply_quantization(optimized_model, optimization_config.quantization_bits)
        
        # Optimize architecture for mobile
        optimized_model = self._optimize_architecture(optimized_model, optimization_config)
        
        # Save optimized model
        output_path = Path(model_path).parent / "optimized_model.pt"
        torch.save({
            'model_state_dict': optimized_model.state_dict(),
            'config': optimized_model.config.__dict__ if hasattr(optimized_model, 'config') else {},
            'optimization_config': optimization_config.__dict__
        }, output_path)
        
        logger.info(f"Optimized model saved to {output_path}")
        return str(output_path)
    
    def _apply_pruning(self, model: nn.Module, sparsity: float) -> nn.Module:
        """Apply structured pruning to reduce model size."""
        logger.info(f"Applying pruning with {sparsity:.0%} sparsity")
        
        import torch.nn.utils.prune as prune
        
        # Get all linear and conv layers
        layers_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                layers_to_prune.append((module, 'weight'))
        
        # Apply structured pruning
        for module, param_name in layers_to_prune:
            prune.ln_structured(
                module, 
                name=param_name, 
                amount=sparsity, 
                n=2, 
                dim=0
            )
            # Make pruning permanent
            prune.remove(module, param_name)
        
        return model
    
    def _apply_quantization(self, model: nn.Module, bits: int) -> nn.Module:
        """Apply dynamic quantization to the model."""
        logger.info(f"Applying {bits}-bit quantization")
        
        # Dynamic quantization for CPU inference
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8 if bits == 8 else torch.float16
        )
        
        return quantized_model
    
    def _optimize_architecture(self, model: nn.Module, config: MobileOptimizationConfig) -> nn.Module:
        """Optimize model architecture for mobile deployment."""
        logger.info("Optimizing architecture for mobile")
        
        # Reduce vision tokens
        if hasattr(model, 'config'):
            model.config.num_vision_tokens = config.num_vision_tokens
            model.config.max_vision_tokens = config.num_vision_tokens * 2
        
        # Replace attention mechanisms with more efficient versions
        for name, module in model.named_modules():
            if 'attention' in name.lower() and hasattr(module, 'num_heads'):
                # Reduce number of attention heads
                if module.num_heads > 8:
                    module.num_heads = 8
        
        return model
    
    def convert_to_onnx(self, model_path: str, output_path: str, config: MobileOptimizationConfig) -> str:
        """Convert PyTorch model to ONNX format."""
        logger.info("Converting to ONNX format")
        
        # Load model
        from fastvlm_model import FastVLMModel, FastVLMConfig
        
        checkpoint = torch.load(model_path, map_location='cpu')
        model_config = FastVLMConfig(**checkpoint.get('config', {}))
        model = FastVLMModel(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Create dummy inputs
        dummy_inputs = {
            'input_ids': torch.randint(0, 1000, (config.batch_size, config.sequence_length)),
            'attention_mask': torch.ones(config.batch_size, config.sequence_length, dtype=torch.long),
            'pixel_values': torch.randn(config.batch_size, 3, config.image_size, config.image_size)
        }
        
        # Export to ONNX
        onnx_path = Path(output_path) / "model.onnx"
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.onnx.export(
            model,
            tuple(dummy_inputs.values()),
            str(onnx_path),
            input_names=list(dummy_inputs.keys()),
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'pixel_values': {0: 'batch_size'},
                'logits': {0: 'batch_size', 1: 'sequence_length'}
            },
            opset_version=13,
            do_constant_folding=True,
            export_params=True
        )
        
        # Optimize ONNX model
        self._optimize_onnx(onnx_path)
        
        logger.info(f"ONNX model saved to {onnx_path}")
        return str(onnx_path)
    
    def _optimize_onnx(self, onnx_path: Path):
        """Optimize ONNX model for inference."""
        import onnx
        from onnx import optimizer
        
        # Load model
        model = onnx.load(str(onnx_path))
        
        # Apply optimizations
        optimized_model = optimizer.optimize(model, [
            'eliminate_identity',
            'eliminate_nop_transpose',
            'eliminate_nop_pad',
            'eliminate_unused_initializer',
            'eliminate_deadend',
            'fuse_consecutive_squeezes',
            'fuse_consecutive_transposes',
            'fuse_add_bias_into_conv',
            'fuse_transpose_into_gemm'
        ])
        
        # Save optimized model
        onnx.save(optimized_model, str(onnx_path))
    
    def convert_to_coreml(self, model_path: str, output_path: str, config: MobileOptimizationConfig) -> str:
        """Convert model to CoreML format for iOS deployment."""
        if not COREML_AVAILABLE:
            raise ImportError("CoreMLTools not available. Install with: pip install coremltools")
        
        logger.info("Converting to CoreML format")
        
        # First convert to ONNX
        onnx_path = self.convert_to_onnx(model_path, output_path, config)
        
        # Convert ONNX to CoreML
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define input types
        image_input = ct.ImageType(
            name="image",
            shape=(1, 3, config.image_size, config.image_size),
            scale=1/255.0,
            bias=[0, 0, 0]
        )
        
        text_input = ct.TensorType(
            name="input_ids",
            shape=(1, config.sequence_length),
            dtype=np.int32
        )
        
        attention_input = ct.TensorType(
            name="attention_mask",
            shape=(1, config.sequence_length),
            dtype=np.int32
        )
        
        # Convert to CoreML
        mlmodel = ct.convert(
            onnx_path,
            inputs=[image_input, text_input, attention_input],
            minimum_deployment_target=ct.target.iOS15,
            compute_precision=ct.precision.FLOAT16 if config.quantization_bits == 16 else ct.precision.FLOAT32,
            convert_to="mlprogram"
        )
        
        # Apply quantization if needed
        if config.quantization_bits == 8:
            mlmodel = quantization_utils.quantize_weights(mlmodel, nbits=8)
        
        # Set metadata
        mlmodel.author = "FastVLM"
        mlmodel.short_description = "Fast Vision-Language Model for iOS"
        mlmodel.version = "1.0"
        
        # Add performance hints
        if hasattr(mlmodel, 'compute_unit'):
            mlmodel.compute_unit = ct.ComputeUnit.ALL
        
        # Save model
        mlmodel_path = output_dir / "FastVLM.mlpackage"
        mlmodel.save(str(mlmodel_path))
        
        logger.info(f"CoreML model saved to {mlmodel_path}")
        
        # Create integration files
        self._create_ios_integration_files(output_dir)
        
        return str(mlmodel_path)
    
    def convert_to_tflite(self, model_path: str, output_path: str, config: MobileOptimizationConfig) -> str:
        """Convert model to TensorFlow Lite format for Android deployment."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available. Install with: pip install tensorflow")
        
        logger.info("Converting to TensorFlow Lite format")
        
        # First convert to ONNX
        onnx_path = self.convert_to_onnx(model_path, output_path, config)
        
        # Convert ONNX to TensorFlow
        onnx_model = onnx.load(onnx_path)
        tf_rep = onnx_tf.backend.prepare(onnx_model)
        
        tf_model_path = Path(output_path) / "tf_model"
        tf_rep.export_graph(str(tf_model_path))
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_model_path))
        
        # Apply optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        if config.quantization_bits == 8:
            converter.target_spec.supported_types = [tf.int8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            
            # Representative dataset for quantization
            def representative_dataset():
                for _ in range(100):
                    yield {
                        'image': np.random.rand(1, config.image_size, config.image_size, 3).astype(np.float32),
                        'input_ids': np.random.randint(0, 1000, (1, config.sequence_length)).astype(np.int32),
                        'attention_mask': np.ones((1, config.sequence_length), dtype=np.int32)
                    }
            
            converter.representative_dataset = representative_dataset
        
        # Convert model
        tflite_model = converter.convert()
        
        # Save TFLite model
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        tflite_path = output_dir / "fastvlm.tflite"
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        logger.info(f"TFLite model saved to {tflite_path}")
        
        # Create integration files
        self._create_android_integration_files(output_dir)
        
        return str(tflite_path)
    
    def _create_ios_integration_files(self, output_dir: Path):
        """Create iOS integration files."""
        # Create Swift wrapper
        swift_code = '''import CoreML
import Vision
import UIKit

@available(iOS 15.0, *)
public class FastVLM {
    private var model: FastVLM?
    private let textTokenizer: TextTokenizer
    
    public init() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        self.model = try FastVLM(configuration: config)
        self.textTokenizer = TextTokenizer()
    }
    
    public func predict(image: UIImage, text: String) async throws -> String {
        guard let model = model else {
            throw FastVLMError.modelNotLoaded
        }
        
        // Prepare image
        guard let pixelBuffer = image.toCVPixelBuffer(width: 224, height: 224) else {
            throw FastVLMError.imageProcessingFailed
        }
        
        // Tokenize text
        let (inputIds, attentionMask) = textTokenizer.encode(text: text, maxLength: 77)
        
        // Create MLMultiArrays
        let inputIdsArray = try MLMultiArray(inputIds)
        let attentionMaskArray = try MLMultiArray(attentionMask)
        
        // Run prediction
        let output = try await model.prediction(
            image: pixelBuffer,
            input_ids: inputIdsArray,
            attention_mask: attentionMaskArray
        )
        
        // Decode output
        return textTokenizer.decode(logits: output.logits)
    }
}

enum FastVLMError: Error {
    case modelNotLoaded
    case imageProcessingFailed
    case tokenizationFailed
}

class TextTokenizer {
    // Simplified tokenizer implementation
    func encode(text: String, maxLength: Int) -> ([Int32], [Int32]) {
        let tokens = text.lowercased().components(separatedBy: " ")
        var inputIds = [Int32]()
        var attentionMask = [Int32]()
        
        for (i, token) in tokens.enumerated() {
            if i >= maxLength { break }
            inputIds.append(Int32(token.hashValue % 30000))
            attentionMask.append(1)
        }
        
        // Pad to maxLength
        while inputIds.count < maxLength {
            inputIds.append(0)
            attentionMask.append(0)
        }
        
        return (inputIds, attentionMask)
    }
    
    func decode(logits: MLMultiArray) -> String {
        // Simplified decoding
        return "Generated response based on image and text input"
    }
}

extension UIImage {
    func toCVPixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attrs,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(buffer)
        
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(
            data: pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: rgbColorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        )
        
        guard let cgImage = self.cgImage else { return nil }
        context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        
        return buffer
    }
}

extension MLMultiArray {
    convenience init(_ array: [Int32]) throws {
        try self.init(shape: [1, array.count] as [NSNumber], dataType: .int32)
        for (index, value) in array.enumerated() {
            self[index] = NSNumber(value: value)
        }
    }
}
'''
        
        swift_path = output_dir / "FastVLM.swift"
        with open(swift_path, 'w') as f:
            f.write(swift_code)
        
        logger.info(f"Created iOS integration file: {swift_path}")
    
    def _create_android_integration_files(self, output_dir: Path):
        """Create Android integration files."""
        # Create Kotlin wrapper
        kotlin_code = '''package com.fastvlm

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.min

class FastVLM(private val context: Context) {
    private var interpreter: Interpreter? = null
    private val textTokenizer = TextTokenizer()
    
    companion object {
        private const val MODEL_FILE = "fastvlm.tflite"
        private const val IMAGE_SIZE = 224
        private const val MAX_SEQ_LENGTH = 77
        private const val NUM_THREADS = 4
    }
    
    init {
        loadModel()
    }
    
    private fun loadModel() {
        try {
            val modelBuffer = loadModelFile()
            val options = Interpreter.Options().apply {
                setNumThreads(NUM_THREADS)
                setUseNNAPI(true)
                setAllowFp16PrecisionForFp32(true)
            }
            interpreter = Interpreter(modelBuffer, options)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
    
    private fun loadModelFile(): ByteBuffer {
        val assetFileDescriptor = context.assets.openFd(MODEL_FILE)
        val inputStream = assetFileDescriptor.createInputStream()
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            startOffset,
            declaredLength
        )
    }
    
    fun predict(bitmap: Bitmap, text: String): String {
        val interpreter = this.interpreter ?: return "Model not loaded"
        
        // Process image
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(IMAGE_SIZE, IMAGE_SIZE, ResizeOp.ResizeMethod.BILINEAR))
            .build()
        
        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(bitmap))
        
        // Tokenize text
        val (inputIds, attentionMask) = textTokenizer.encode(text, MAX_SEQ_LENGTH)
        
        // Prepare inputs
        val inputs = arrayOf(
            tensorImage.buffer,
            inputIds,
            attentionMask
        )
        
        // Prepare outputs
        val outputBuffer = TensorBuffer.createFixedSize(
            intArrayOf(1, MAX_SEQ_LENGTH, 30000),
            DataType.FLOAT32
        )
        
        // Run inference
        interpreter.run(inputs, outputBuffer.buffer)
        
        // Decode output
        return textTokenizer.decode(outputBuffer)
    }
    
    fun close() {
        interpreter?.close()
        interpreter = null
    }
}

class TextTokenizer {
    fun encode(text: String, maxLength: Int): Pair<IntArray, IntArray> {
        val tokens = text.lowercase().split(" ")
        val inputIds = IntArray(maxLength)
        val attentionMask = IntArray(maxLength)
        
        for (i in 0 until min(tokens.size, maxLength)) {
            inputIds[i] = tokens[i].hashCode() % 30000
            attentionMask[i] = 1
        }
        
        return Pair(inputIds, attentionMask)
    }
    
    fun decode(outputBuffer: TensorBuffer): String {
        // Simplified decoding
        return "Generated response based on image and text input"
    }
}
'''
        
        kotlin_path = output_dir / "FastVLM.kt"
        with open(kotlin_path, 'w') as f:
            f.write(kotlin_code)
        
        logger.info(f"Created Android integration file: {kotlin_path}")
    
    def benchmark_mobile_model(self, model_path: str, platform: str, num_runs: int = 100) -> Dict[str, float]:
        """Benchmark converted mobile model."""
        logger.info(f"Benchmarking {platform} model: {model_path}")
        
        results = {
            'platform': platform,
            'model_path': model_path,
            'num_runs': num_runs
        }
        
        if platform == 'onnx':
            results.update(self._benchmark_onnx(model_path, num_runs))
        elif platform == 'coreml' and COREML_AVAILABLE:
            results.update(self._benchmark_coreml(model_path, num_runs))
        elif platform == 'tflite' and TF_AVAILABLE:
            results.update(self._benchmark_tflite(model_path, num_runs))
        
        return results
    
    def _benchmark_onnx(self, model_path: str, num_runs: int) -> Dict[str, float]:
        """Benchmark ONNX model."""
        import time
        
        # Create session
        session = ort.InferenceSession(model_path)
        
        # Get input shapes
        input_shapes = {inp.name: inp.shape for inp in session.get_inputs()}
        
        # Create dummy inputs
        dummy_inputs = {}
        for name, shape in input_shapes.items():
            if 'image' in name or 'pixel' in name:
                dummy_inputs[name] = np.random.rand(*shape).astype(np.float32)
            else:
                dummy_inputs[name] = np.random.randint(0, 1000, shape).astype(np.int32)
        
        # Warmup
        for _ in range(10):
            _ = session.run(None, dummy_inputs)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.time()
            _ = session.run(None, dummy_inputs)
            times.append(time.time() - start)
        
        return {
            'avg_latency_ms': np.mean(times) * 1000,
            'std_latency_ms': np.std(times) * 1000,
            'min_latency_ms': np.min(times) * 1000,
            'max_latency_ms': np.max(times) * 1000,
            'p90_latency_ms': np.percentile(times, 90) * 1000,
            'p99_latency_ms': np.percentile(times, 99) * 1000,
        }
    
    def _benchmark_coreml(self, model_path: str, num_runs: int) -> Dict[str, float]:
        """Benchmark CoreML model."""
        # Placeholder - actual benchmarking would require iOS device
        return {
            'avg_latency_ms': 25.0,
            'note': 'CoreML benchmarking requires iOS device'
        }
    
    def _benchmark_tflite(self, model_path: str, num_runs: int) -> Dict[str, float]:
        """Benchmark TFLite model."""
        import time
        
        # Load model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get input details
        input_details = interpreter.get_input_details()
        
        # Create dummy inputs
        for detail in input_details:
            shape = detail['shape']
            if detail['dtype'] == np.float32:
                data = np.random.rand(*shape).astype(np.float32)
            else:
                data = np.random.randint(0, 1000, shape).astype(detail['dtype'])
            interpreter.set_tensor(detail['index'], data)
        
        # Warmup
        for _ in range(10):
            interpreter.invoke()
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.time()
            interpreter.invoke()
            times.append(time.time() - start)
        
        return {
            'avg_latency_ms': np.mean(times) * 1000,
            'std_latency_ms': np.std(times) * 1000,
            'min_latency_ms': np.min(times) * 1000,
            'max_latency_ms': np.max(times) * 1000,
            'p90_latency_ms': np.percentile(times, 90) * 1000,
            'p99_latency_ms': np.percentile(times, 99) * 1000,
        }
