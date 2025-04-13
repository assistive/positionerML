# RNN-Based IMU Data Smoothing Guide

This document provides a detailed guide on the implementation and usage of the recurrent neural network (RNN) smoothing feature in the Vehicle Motion Tracking System.

## Overview

The RNN-based smoothing system uses deep learning to filter noise from IMU sensor data while preserving important motion events. Unlike traditional filters (like low-pass or Kalman filters), the RNN approach can learn complex patterns in the data, allowing it to distinguish between sensor noise and actual vehicle motion.

## How It Works

1. **Neural Network Architecture**: 
   - A bidirectional LSTM (Long Short-Term Memory) network processes sequences of IMU data
   - Multiple layers capture different temporal patterns and relationships
   - Trained to map noisy sensor readings to clean, smoothed outputs

2. **Real-time Processing**:
   - The system maintains a buffer of incoming IMU data
   - Data is processed in small batches to maintain responsiveness
   - TensorFlow Lite runs the inference efficiently on the mobile device

3. **Sequence-to-Sequence Approach**:
   - Each IMU sequence is processed as a whole
   - The model considers context from surrounding data points
   - Output maintains the same shape and timing as the input

## Key Benefits

- **Preserves Motion Events**: Unlike simple filters, the RNN maintains sharp transitions during significant motion events (acceleration, braking, cornering)
- **Adaptive Smoothing**: Applies appropriate smoothing based on the context of the motion
- **Improved Event Detection**: Reduces false positives by removing noise that might trigger event thresholds
- **Multi-channel Processing**: Processes all IMU channels simultaneously, preserving relationships between axes

## Technical Implementation

### Components

1. **IMUSmoother Class**:
   - Loads the TensorFlow Lite model
   - Handles data preprocessing and batching
   - Provides methods for smoothing both batch and streaming data

2. **Model Training Scripts**:
   - Python scripts for training the RNN model
   - Includes synthetic data generation for training
   - Exports to TensorFlow Lite format

3. **Integration with MotionTracker**:
   - Seamless integration with the existing data pipeline
   - Maintains both raw and smoothed data streams
   - Configuration options to enable/disable smoothing

### File Structure

- `shared/src/androidMain/kotlin/com/assistive/vehiclemotiontracking/model/IMUSmoother.kt`: Android implementation
- `shared/src/iosMain/kotlin/com/assistive/vehiclemotiontracking/model/IMUSmoother.kt`: iOS stub implementation
- `app/src/main/assets/imu_smoother.tflite`: TensorFlow Lite model file

## Using the RNN Smoothing Feature

### Configuration

The RNN smoothing can be configured in the `Configuration` class:

```kotlin
val config = Configuration(
    samplingRateHz = 200,
    bufferWindowSeconds = 60,
    useRnnSmoothing = true,    // Enable/disable RNN smoothing
    rnnBatchSize = 50          // Batch size for processing
)

val motionTracker = MotionTracker(platformServices, config)
```

### Accessing Smoothed Data

The MotionTracker provides access to both raw and smoothed data:

```kotlin
// Raw data flow
lifecycleScope.launch {
    motionTracker.motionDataFlow.collect { dataPoint ->
        // Process raw data
    }
}

// Smoothed data flow
lifecycleScope.launch {
    motionTracker.smoothedDataFlow.collect { dataPoint ->
        // Process smoothed data
    }
}

// Get current buffer content
val rawBuffer = motionTracker.getBufferData()
val smoothedBuffer = motionTracker.getSmoothedBufferData()
```

### Visualization

The Compose-based visualization screen allows you to compare raw and smoothed data in real-time:

1. Launch the app and start tracking
2. Tap the "Open Data Visualization" button
3. The visualization screen will show both raw (blue) and smoothed (green) acceleration data

### Exporting Data

When you export data using the "Save Current 60s Buffer" button, both raw and smoothed data are included in the CSV:

```
timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,v_accel_x,v_accel_y,v_accel_z,speed,is_accel,is_brake,is_corner,smoothed_accel_x,smoothed_accel_y,smoothed_accel_z,smoothed_gyro_x,smoothed_gyro_y,smoothed_gyro_z,...
```

## Training Custom Models

You can train custom models for specific vehicles or sensors:

1. Collect IMU data from your target vehicle
2. Use the provided Python script to train a custom model:

```bash
python train_rnn_model.py --data your_data.csv --epochs 30 --batch_size 32
```

3. The script will generate a `imu_smoother.tflite` file
4. Replace the existing model in the assets folder with your custom model

## Performance Considerations

- **Processing Time**: The RNN inference adds some processing overhead. On most modern phones, this is negligible.
- **Battery Impact**: The additional processing can increase battery consumption by 5-10%.
- **Memory Usage**: The model requires approximately 1MB of memory.

## Troubleshooting

- **Model Loading Failures**: If the TensorFlow Lite model fails to load, the system will fall back to pass-through mode (no smoothing).
- **High Latency**: If you experience high latency, try reducing the sampling rate or increasing the batch size.
- **Poor Smoothing Quality**: This could indicate a mismatch between the training data and your current usage. Consider training a custom model.

## Future Improvements

- Quantized models for better performance
- Online learning to adapt to specific vehicles
- More sophisticated network architectures
- iOS implementation using CoreML
