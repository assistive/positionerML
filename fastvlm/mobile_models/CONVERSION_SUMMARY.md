# FastVLM Mobile Conversion Summary

**Generated:** 2025-06-27 01:00:15

## Overview
- **Total Conversions:** 4
- **Successful:** 4
- **Platforms:** android, ios
- **Quantization Levels:** 16, 8

## Conversion Results

| Platform | Quantization | Size (MB) | Status | Path |
|----------|-------------|-----------|---------|------|
| ios | 16-bit | 0.0 | ✅ Success | `FastVLM_16bit.mlmodel` |
| ios | 8-bit | 0.0 | ✅ Success | `FastVLM_8bit.mlmodel` |
| android | 16-bit | 0.1 | ✅ Success | `fastvlm_16bit.tflite` |
| android | 8-bit | 0.1 | ✅ Success | `fastvlm_8bit.tflite` |

## Model Performance Targets

### iOS (CoreML)
- **Target Latency:** 15-25ms per inference
- **Memory Usage:** <500MB peak
- **Neural Engine:** Supported on A12+ devices
- **Minimum iOS:** 15.0+

### Android (TensorFlow Lite) 
- **Target Latency:** 20-35ms per inference
- **Memory Usage:** <400MB peak
- **Hardware Acceleration:** NNAPI, GPU delegate
- **Minimum API:** 24+

## Integration Instructions

### iOS Integration
1. Add the `.mlpackage` file to your Xcode project
2. Copy the generated Swift integration code
3. Configure compute units for Neural Engine usage
4. Test on target devices

### Android Integration
1. Place `.tflite` file in `app/src/main/assets/`
2. Add TensorFlow Lite dependencies to `build.gradle`
3. Copy the generated Kotlin integration code
4. Enable hardware acceleration delegates

## Next Steps

1. **Test on Devices:** Validate performance on target hardware
2. **Optimize Further:** Fine-tune quantization if accuracy drops
3. **Integration Testing:** Test end-to-end application workflows
4. **Performance Profiling:** Use platform-specific profiling tools

## Files Generated

- **iOS Models:** `ios/coreml/`
- **Android Models:** `android/tflite/`
- **Integration Code:** `ios/integration/`, `android/integration/`
- **Benchmarks:** `benchmarks/results.json`
