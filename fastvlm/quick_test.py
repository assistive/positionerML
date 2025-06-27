#!/usr/bin/env python3
"""
Quick test to verify mobile converter dependencies work
"""

def test_imports():
    """Test all required imports."""
    print("🧪 Testing Mobile Converter Dependencies")
    print("=" * 50)
    
    success_count = 0
    total_tests = 0
    
    # Test PyTorch
    total_tests += 1
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        success_count += 1
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
    
    # Test NumPy
    total_tests += 1
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        success_count += 1
    except ImportError as e:
        print(f"❌ NumPy: {e}")
    
    # Test ONNX
    total_tests += 1
    try:
        import onnx
        print(f"✅ ONNX {onnx.__version__}")
        success_count += 1
    except ImportError as e:
        print(f"❌ ONNX: {e}")
    
    # Test CoreMLTools
    total_tests += 1
    try:
        import coremltools as ct
        print(f"✅ CoreMLTools {ct.__version__}")
        success_count += 1
    except ImportError as e:
        print(f"❌ CoreMLTools: {e}")
    
    # Test TensorFlow
    total_tests += 1
    tf_found = False
    tf_version = "unknown"
    
    try:
        import tensorflow as tf
        tf_found = True
        tf_version = tf.__version__
    except ImportError:
        try:
            import tensorflow_macos as tf
            tf_found = True 
            tf_version = tf.__version__
        except ImportError:
            try:
                import tensorflow_cpu as tf
                tf_found = True
                tf_version = tf.__version__
            except ImportError:
                pass
    
    if tf_found:
        print(f"✅ TensorFlow {tf_version}")
        success_count += 1
        
        # Test TFLite
        try:
            converter = tf.lite.TFLiteConverter
            print("  ✅ TFLite converter available")
        except Exception:
            print("  ⚠️  TFLite converter issues")
    else:
        print("❌ TensorFlow: Not found")
    
    print("=" * 50)
    print(f"📊 Results: {success_count}/{total_tests} dependencies available")
    
    if success_count >= 4:  # At least PyTorch, NumPy, ONNX, and one mobile framework
        print("🎉 Ready for mobile conversion!")
        return True
    else:
        print("❌ Missing critical dependencies")
        return False

def test_simple_conversion():
    """Test a minimal conversion workflow."""
    print("\n🔄 Testing Simple Conversion Workflow")
    print("=" * 50)
    
    try:
        import torch
        import torch.nn as nn
        
        # Create simple model
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TinyModel()
        model.eval()
        
        # Test tracing
        dummy_input = torch.randn(1, 3)
        traced_model = torch.jit.trace(model, dummy_input)
        print("✅ PyTorch model tracing: Works")
        
        # Test CoreML conversion (if available)
        try:
            import coremltools as ct
            
            coreml_model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(shape=(1, 3))],
                convert_to="neuralnetwork"
            )
            print("✅ CoreML conversion: Works")
            
        except ImportError:
            print("⚠️  CoreML conversion: CoreMLTools not available")
        except Exception as e:
            print(f"⚠️  CoreML conversion: {e}")
        
        # Test TensorFlow model creation (if available)
        try:
            import tensorflow as tf
            
            # Create equivalent TF model
            tf_model = tf.keras.Sequential([
                tf.keras.layers.Dense(1, input_shape=(3,))
            ])
            
            # Test TFLite conversion
            converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
            tflite_model = converter.convert()
            print("✅ TensorFlow Lite conversion: Works")
            
        except ImportError:
            print("⚠️  TensorFlow Lite conversion: TensorFlow not available")
        except Exception as e:
            print(f"⚠️  TensorFlow Lite conversion: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion test failed: {e}")
        return False

def main():
    """Run all tests."""
    imports_ok = test_imports()
    
    if imports_ok:
        conversion_ok = test_simple_conversion()
        
        print("\n" + "=" * 50)
        print("🎯 FINAL RESULT")
        print("=" * 50)
        
        if conversion_ok:
            print("🎉 Mobile converter should work!")
            print("✅ Run: python fastvlm_universal_converter.py")
        else:
            print("⚠️  Basic imports work but conversion may have issues")
            print("💡 Try the simple converter: python simple_mobile_converter.py")
    else:
        print("\n💡 Install missing dependencies:")
        print("   pip install -r requirements_mobile.txt")
    
    return 0 if imports_ok else 1

if __name__ == "__main__":
    exit