# TCN Implementation Plan for IMU Dead Reckoning

## 1. Introduction

This document outlines the plan to implement a Temporal Convolutional Network (TCN) for IMU-based dead reckoning and data smoothing. TCNs offer several advantages over RNNs for this application:

- **Parallelizable computation**: Unlike sequential RNNs, TCNs process the entire sequence at once
- **Stable gradients**: No vanishing/exploding gradient problems that plague RNNs
- **Flexible receptive field**: Can easily be adjusted to capture long-range dependencies
- **Constant memory usage**: Independent of sequence length
- **Potentially higher accuracy**: Recent research suggests TCNs outperform RNNs in many sequence tasks

## 2. TCN Architecture

### Core Components

1. **Dilated Causal Convolutions**:
   - Causal padding ensures no information leakage from future to past
   - Dilation expands the receptive field exponentially with depth
   - Multiple stacked layers to capture patterns at different time scales

2. **Residual Blocks**:
   - Each block contains two dilated convolutional layers
   - Weight normalization and dropout for regularization
   - Residual connections to improve gradient flow

3. **Model Architecture**:
   - Input: Sequence of IMU data (acc_x/y/z, gyro_x/y/z, etc.)
   - Hidden layers: Stack of residual blocks with increasing dilation rates
   - Output: Either smoothed IMU values or position/velocity deltas

### Implementation Details

```python
class ResidualBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super(ResidualBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                                  padding=padding, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                                  padding=padding, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                               self.conv2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNModel(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=3, dropout=0.2):
        super(TCNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [ResidualBlock(in_channels, out_channels, kernel_size, dilation=dilation_size, dropout=dropout)]
        
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)
        
    def forward(self, x):
        # x needs to have shape (batch_size, channels, seq_len)
        batch_size, seq_len, n_features = x.size()
        x = x.transpose(1, 2)  # (batch_size, n_features, seq_len)
        
        # Apply the TCN layers
        y = self.network(x)
        
        # Global average pooling or use the last time step
        # y = y.mean(dim=2)  # Global average pooling
        y = y[:, :, -1]  # Last time step
        
        # Project to output size
        return self.linear(y)
```

## 3. Data Preparation

### Data Requirements

- **IMU sequences**: Continuous streams of accelerometer and gyroscope readings
- **Ground truth positions**: For supervised learning of dead reckoning
- **Clean IMU data**: For training the smoothing model

### Preprocessing Steps

1. **Sequence creation**:
   - Sliding window approach with appropriate overlap
   - Window size based on expected temporal dependencies (50-200 samples)
   
2. **Feature engineering**:
   - Calculate magnitude of acceleration and angular velocity
   - Compute jerk (derivative of acceleration)
   - Add sensor bias and noise features if available
   
3. **Normalization**:
   - Standardize all features to zero mean and unit variance
   - Save normalization parameters for inference

4. **Data augmentation**:
   - Add random noise to simulate sensor inaccuracies
   - Apply small rotations to account for different device orientations
   - Simulate different walking/driving styles

## 4. Training Pipeline

### Training Strategy

1. **Loss functions**:
   - For smoothing: Mean Squared Error (MSE) between predicted and clean IMU data
   - For dead reckoning: Weighted MSE for position and velocity deltas

2. **Training regime**:
   - Batch size: 32-64 sequences
   - Learning rate: 0.001 with decay
   - Optimizer: Adam with weight decay
   - Early stopping based on validation loss

3. **Hyperparameter tuning**:
   - Kernel size: Typically 2-5
   - Number of channels: [32, 64, 128] or larger
   - Dilation rates: Typically powers of 2 [1, 2, 4, 8, 16, ...]
   - Dropout rate: 0.2-0.5

### Evaluation Metrics

1. **Smoothing**:
   - Mean Absolute Error (MAE)
   - Power Spectral Density comparison
   - Signal-to-noise ratio improvement

2. **Dead Reckoning**:
   - Absolute Trajectory Error (ATE)
   - Relative Position Error (RPE)
   - Maximum position error during simulated GPS outages

## 5. Public Datasets for Training

1. **Oxford Inertial Odometry Dataset (OxIOD)**
   - High-quality IMU data with ground truth positions
   - Multiple motion scenarios (walking, running, stair climbing)
   - Various device placements and orientations

2. **RIDI Dataset**
   - Smartphone IMU data with ground truth trajectories
   - Indoor scenarios with different motion patterns
   - Good for testing robustness in challenging environments

3. **RoNIN Dataset**
   - Large-scale dataset for inertial navigation
   - Diverse set of users and motion scenarios
   - Ground truth from visual-inertial odometry

4. **KITTI Dataset**
   - Vehicle motion with synchronized IMU and GPS
   - Outdoor environments with varying dynamics
   - Useful for automotive applications

5. **KAIST Urban Dataset**
   - Multi-modal urban driving dataset
   - Contains IMU, GPS, and LiDAR data
   - Good for testing in complex urban environments

## 6. Implementation Timeline

### Phase 1: Setup and Data Preparation (1-2 weeks)
- Set up development environment
- Download and preprocess selected datasets
- Implement data loading and augmentation pipeline

### Phase 2: TCN Model Implementation (1-2 weeks)
- Implement the TCN architecture
- Create training and evaluation loops
- Develop visualization tools for model performance

### Phase 3: Model Training and Optimization (2-3 weeks)
- Train initial models on selected datasets
- Perform hyperparameter tuning
- Compare performance against RNN baselines

### Phase 4: Evaluation and Fine-tuning (1-2 weeks)
- Evaluate on test datasets
- Analyze failure cases
- Fine-tune models for specific applications

### Phase 5: Mobile Deployment (2-3 weeks)
- Optimize model for mobile deployment
- Convert to TensorFlow Lite / CoreML
- Implement real-time processing pipeline

## 7. Implementation Code Structure

```
/tcn/
  /data/
    data_loader.py      # Dataset loading and preprocessing
    augmentation.py     # Data augmentation techniques
  /models/
    tcn_model.py        # TCN model architecture
    smoothing_model.py  # TCN for IMU smoothing
    dead_reckoning.py   # TCN for dead reckoning
  /training/
    trainer.py          # Training loop and logging
    losses.py           # Custom loss functions
    metrics.py          # Evaluation metrics
  /evaluation/
    visualizer.py       # Result visualization
    comparator.py       # Comparison with baselines
  /deployment/
    converter.py        # Model conversion for deployment
    realtime_processor.py  # Real-time processing pipeline
  main.py               # Main entry point
  config.py             # Configuration parameters
```

## 8. Comparison with Existing RNN Approach

### Advantages of TCN
- Parallelizable computation (faster training)
- Fixed memory usage regardless of sequence length
- More stable gradients leading to better convergence
- Explicit control over the receptive field size

### Potential Challenges
- May require more parameters for equivalent receptive field
- Could be less efficient for very long sequences
- Less intuitive interpretation compared to RNNs

## 9. Extensions and Future Work

1. **Attention mechanisms**:
   - Add attention to focus on important parts of the input sequence
   - Implement self-attention or cross-attention between modalities

2. **Uncertainty estimation**:
   - Output confidence intervals for position estimates
   - Implement Monte Carlo dropout for uncertainty quantification

3. **Transfer learning**:
   - Pre-train on large datasets and fine-tune for specific devices
   - Develop domain adaptation techniques for different environments

4. **Multi-modal fusion**:
   - Incorporate additional sensors (magnetometer, barometer)
   - Develop sensor fusion mechanisms with GPS when available
