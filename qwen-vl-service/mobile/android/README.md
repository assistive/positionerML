# Android Deployment for Qwen 2.5-VL

This directory contains the Android deployment setup for Qwen 2.5-VL models.

## Requirements

- Android Studio 2023.1+
- Android API 24+ (Android 7.0+)
- Device with 6GB+ RAM
- Preferably device with NNAPI support

## Setup

1. Convert model to TensorFlow Lite:
   ```bash
   python scripts/convert_mobile.py qwen-2.5-vl-3b --platform android
   ```

2. Add the generated `.tflite` file to your Android project's assets

3. Use the Kotlin integration code in the `android_integration/` directory

## Performance

- **Model**: Qwen 2.5-VL-3B  
- **Size**: ~6GB (quantized)
- **Inference**: 200-800ms per request
- **Memory**: 4-6GB peak usage
