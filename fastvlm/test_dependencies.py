#!/usr/bin/env python3
"""
Test script to verify all dependencies are working correctly
"""

def test_pytorch():
    """Test PyTorch availability."""
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} - Available")
        
        # Test basic operations
        x = torch.randn(2, 3)
        y = torch.randn(3, 4)
        z = torch.mm(x, y)
        print(f"   Basic operations: Working")
        
        return True
    except ImportError as e:
        print(f"❌ PyTorch - Not available: {e}")
        return False

def test_coremltools():
    """Test CoreMLTools availability."""
    try:
        import coremltools as ct
        print(f"✅ CoreMLTools {ct.__version__} - Available")
        
        # Test basic functionality
        from coremltools.models.neural_network import NeuralNetworkBuilder
        print(f"   Neural network builder: Working")
        
        return True
    except ImportError as e:
        print(f"❌ CoreMLTools - Not available: {e}")
        return False

def test_tensorflow():
    """Test TensorFlow availability with multiple import methods."""
    # Try standard tensorflow
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} - Available")
        test_tf_operations(tf)
        return True
    except ImportError:
        pass
    
    # Try tensorflow-macos (Apple Silicon)
    try:
        import tensorflow_macos as tf
        print(f"✅ TensorFlow-macOS {tf.__version__} - Available")
        test_tf_operations(tf)
        return True
    except ImportError:
        pass
    
    # Try tensorflow-cpu
    try:
        import tensorflow_cpu as tf
        print(f"✅ TensorFlow-CPU {tf.__version__} - Available")
        test_tf_operations(tf)
        return True
    except ImportError:
        pass
    
    print(f"❌ TensorFlow - Not available with any import method")
    print(f"   Tried: tensorflow, tensorflow_macos, tensorflow_cpu")
    
    # Check what TF packages are actually installed
    import subprocess
    import sys
    
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                              capture_output=True, text=True)
        tf_packages = [line for line in result.stdout.split('\n') 
                      if 'tensorflow' in line.lower()]
        if tf_packages:
            print(f"   Detected TensorFlow packages:")
            for pkg in tf_packages:
                print(f"     {pkg}")
        else:
            print(f"   No TensorFlow packages found in pip list")
    except:
        pass
    
    return False

def test_tf_operations(tf):
    """Test TensorFlow operations."""
    try:
        # Test basic operations
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        y = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        z = tf.matmul(x, y)
        print(f"   Basic operations: Working")
        
        # Test TFLite converter
        try:
            converter = tf.lite.TFLiteConverter
            print(f"   TFLite converter: Available")
        except Exception as e:
            print(f"   TFLite converter: Error - {e}")
    except Exception as e:
        print(f"   Basic operations: Error - {e}")

def test_onnx():
    """Test ONNX availability."""
    try:
        import onnx
        print(f"✅ ONNX {onnx.__version__} - Available")
        
        # Test onnx-tf bridge
        try:
            import onnx_tf
            print(f"   ONNX-TensorFlow bridge: Available")
        except ImportError:
            print(f"   ONNX-TensorFlow bridge: Not available (optional)")
        
        return True
    except ImportError as e:
        print(f"❌ ONNX - Not available: {e}")
        return False

def test_numpy():
    """Test NumPy availability."""
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__} - Available")
        
        # Test basic operations
        x = np.random.randn(3, 3)
        y = np.dot(x, x.T)
        print(f"   Basic operations: Working")
        
        return True
    except ImportError as e:
        print(f"❌ NumPy - Not available: {e}")
        return False

def create_simple_models():
    """Test creating simple models."""
    print("\n🧪 Testing Model Creation:")
    
    # Test PyTorch model
    try:
        import torch
        import torch.nn as nn
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        x = torch.randn(1, 10)
        output = model(x)
        print("✅ PyTorch model creation: Working")
        
    except Exception as e:
        print(f"❌ PyTorch model creation: Failed - {e}")
    
    # Test TensorFlow model
    tf = None
    try:
        import tensorflow as tf
        tf_available = True
    except ImportError:
        try:
            import tensorflow_macos as tf
            tf_available = True
        except ImportError:
            try:
                import tensorflow_cpu as tf
                tf_available = True
            except ImportError:
                tf_available = False
    
    if tf_available and tf is not None:
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(1, input_shape=(10,))
            ])
            
            x = tf.random.normal((1, 10))
            output = model(x)
            print("✅ TensorFlow model creation: Working")
            
            # Test TFLite conversion
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            print("✅ TFLite conversion: Working")
            
        except Exception as e:
            print(f"❌ TensorFlow model creation: Failed - {e}")
    else:
        print("❌ TensorFlow not available for model testing")

def main():
    """Run all dependency tests."""
    print("🔍 FastVLM Mobile Converter - Dependency Test")
    print("=" * 60)
    
    tests = [
        test_numpy,
        test_pytorch, 
        test_coremltools,
        test_tensorflow,
        test_onnx
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    # Test model creation
    create_simple_models()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 DEPENDENCY SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print("🎉 All dependencies are working correctly!")
        print("✅ Ready for mobile conversion")
    elif passed >= 3:  # At least PyTorch, NumPy, and one mobile framework
        print(f"⚠️  {passed}/{total} dependencies working")
        print("🟡 Partial functionality available")
    else:
        print(f"❌ Only {passed}/{total} dependencies working")
        print("🔴 Mobile conversion may not work properly")
    
    print("\n💡 Recommendations:")
    if not any([test_coremltools(), test_tensorflow()]):
        print("   - Install CoreMLTools: pip install coremltools")
        print("   - Install TensorFlow: pip install tensorflow")
    elif not test_coremltools():
        print("   - Install CoreMLTools for iOS: pip install coremltools")
    elif not test_tensorflow():
        print("   - Install TensorFlow for Android: pip install tensorflow")
    else:
        print("   - All essential dependencies are installed!")
    
    return 0 if passed >= 3 else 1

if __name__ == "__main__":
    exit(main())
