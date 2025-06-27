#!/usr/bin/env python3
"""
Quick test script for mobile conversion dependencies
"""

def test_dependencies():
    print("🧪 Testing FastVLM Mobile Conversion Dependencies")
    print("=" * 50)
    
    # Test PyTorch
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not available")
        return False
    
    # Test ONNX
    try:
        import onnx
        print(f"✅ ONNX {onnx.__version__}")
    except ImportError:
        print("❌ ONNX not available")
        return False
    
    # Test CoreMLTools
    try:
        import coremltools as ct
        print(f"✅ CoreMLTools {ct.__version__}")
        ios_available = True
    except ImportError:
        print("⚠️  CoreMLTools not available - iOS conversion disabled")
        ios_available = False
    
    # Test TensorFlow
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__}")
        android_available = True
    except ImportError:
        print("⚠️  TensorFlow not available - Android conversion disabled")
        android_available = False
    
    print(f"\n📱 Platform Support:")
    print(f"   iOS (CoreML): {'✅ Available' if ios_available else '❌ Not available'}")
    print(f"   Android (TFLite): {'✅ Available' if android_available else '❌ Not available'}")
    
    if ios_available or android_available:
        print("\n🎉 Ready for mobile conversion!")
        return True
    else:
        print("\n❌ No mobile platforms available")
        return False

if __name__ == "__main__":
    test_dependencies()
