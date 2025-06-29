#!/usr/bin/env python3
"""
MobileCLIP Mobile Benchmarking
Performance testing for mobile deployments
"""
import argparse
import time
import logging
import sys
from pathlib import Path
import json
import numpy as np

def benchmark_ios_model(model_path: str, num_runs: int = 10):
    """Benchmark iOS CoreML model."""
    try:
        import coremltools as ct
    except ImportError:
        print("❌ CoreML Tools not available")
        return None
    
    try:
        # Load model
        model = ct.models.MLModel(model_path)
        
        # Create test input
        test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        # Warmup
        for _ in range(3):
            _ = model.predict({'image': test_input})
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = model.predict({'image': test_input})
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return {
            'platform': 'ios',
            'model_path': model_path,
            'avg_inference_ms': np.mean(times),
            'std_inference_ms': np.std(times),
            'min_inference_ms': np.min(times),
            'max_inference_ms': np.max(times),
            'num_runs': num_runs
        }
        
    except Exception as e:
        print(f"❌ iOS benchmark failed: {e}")
        return None

def benchmark_android_model(model_path: str, num_runs: int = 10):
    """Benchmark Android TensorFlow Lite model."""
    try:
        import tensorflow as tf
    except ImportError:
        print("❌ TensorFlow not available")
        return None
    
    try:
        # Load model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Create test input
        input_shape = input_details[0]['shape']
        test_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(3):
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
            _ = interpreter.get_tensor(output_details[0]['index'])
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
            _ = interpreter.get_tensor(output_details[0]['index'])
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return {
            'platform': 'android',
            'model_path': model_path,
            'avg_inference_ms': np.mean(times),
            'std_inference_ms': np.std(times),
            'min_inference_ms': np.min(times),
            'max_inference_ms': np.max(times),
            'num_runs': num_runs,
            'model_size_mb': Path(model_path).stat().st_size / (1024 * 1024)
        }
        
    except Exception as e:
        print(f"❌ Android benchmark failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Benchmark MobileCLIP mobile models")
    parser.add_argument("--model-path", required=True, help="Path to model file")
    parser.add_argument("--platform", choices=["ios", "android"], required=True, help="Target platform")
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark runs")
    parser.add_argument("--output", help="Output JSON file for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)
    
    print(f"🔍 Benchmarking {args.platform.upper()} model: {model_path}")
    print(f"🏃 Running {args.runs} benchmark iterations...")
    print()
    
    # Run benchmark
    if args.platform == "ios":
        results = benchmark_ios_model(str(model_path), args.runs)
    elif args.platform == "android":
        results = benchmark_android_model(str(model_path), args.runs)
    else:
        print(f"❌ Unsupported platform: {args.platform}")
        sys.exit(1)
    
    if results is None:
        print("❌ Benchmark failed")
        sys.exit(1)
    
    # Print results
    print("📊 BENCHMARK RESULTS")
    print("="*40)
    print(f"Platform: {results['platform'].upper()}")
    print(f"Model: {Path(results['model_path']).name}")
    print(f"Average inference time: {results['avg_inference_ms']:.2f} ms")
    print(f"Standard deviation: {results['std_inference_ms']:.2f} ms")
    print(f"Min inference time: {results['min_inference_ms']:.2f} ms")
    print(f"Max inference time: {results['max_inference_ms']:.2f} ms")
    
    if 'model_size_mb' in results:
        print(f"Model size: {results['model_size_mb']:.2f} MB")
    
    print(f"Number of runs: {results['num_runs']}")
    print("="*40)
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: {args.output}")

if __name__ == "__main__":
    main()
