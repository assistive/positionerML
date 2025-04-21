# TCN Project Plan for IMU-Based Dead Reckoning

## 1. Introduction

This document outlines the project plan for implementing Temporal Convolutional Networks (TCNs) for IMU-based dead reckoning during GPS outages. Building on our existing RNN/LSTM implementation, we aim to leverage TCNs to improve position tracking accuracy and computational efficiency.

## 2. Background and Motivation

### Why TCNs for Dead Reckoning?

The current RNN/LSTM approach has shown promising results for dead reckoning, but TCNs offer several compelling advantages:

- **Parallelization**: Unlike RNNs that process data sequentially, TCNs can process the entire sequence in parallel, offering significant speedup during both training and inference.
- **Fixed Receptive Field**: TCNs provide precise control over the temporal receptive field, allowing explicit modeling of dependencies over specific time periods.
- **Stable Gradients**: TCNs don't suffer from vanishing/exploding gradients that can affect RNNs, enabling more stable training on long sequences.
- **Lower Memory Usage**: Efficient dilated convolutions reduce the memory footprint during training.
- **Computational Efficiency**: TCNs are often more computationally efficient on mobile devices compared to RNNs/LSTMs.

## 3. Technical Approach

### 3.1 Model Architecture

Our TCN architecture for dead reckoning will include:

1. **Input Layer**: Accept IMU sequences (accelerometer + gyroscope) and initial state.
2. **Feature Extraction Layers**: 
   - 1D Causal Convolutions with dilated filters 
   - Increasing dilation factors (1, 2, 4, 8, 16...)
   - Residual connections for gradient flow
3. **Regression Heads**:
   - Position delta prediction (dx, dy, dz)
   - Velocity delta prediction (dvx, dvy, dvz)

### 3.2 Key TCN Components

The core building blocks include:

#### Causal Convolutions
```
       Input
         |
    [Padding]
         |
  [Conv1D with d=1]
         |
     Output
```

#### Dilated Convolutions
Dilated convolutions expand the receptive field exponentially with depth:
- Layer 1: dilation=1 (receptive field: 3)
- Layer 2: dilation=2 (receptive field: 7)
- Layer 3: dilation=4 (receptive field: 15)
- Layer 4: dilation=8 (receptive field: 31)

#### Residual Blocks
```
     Input
    /     \
 [TCN]    |
    \     /
      [+]
       |
   [Output]
```

### 3.3 Loss Functions

We will use a composite loss function:
- MSE loss for position deltas
- MSE loss for velocity deltas
- Optional: Physics-based constraints from kinematic equations

## 4. Implementation Plan

### 4.1 Phase 1: Core Architecture (Weeks 1-2)

- Implement basic TCN architecture in TensorFlow/Keras and PyTorch
- Design the residual block structure
- Implement dilation pattern for optimal temporal receptive field
- Create data preprocessing pipeline compatible with TCN input requirements

### 4.2 Phase 2: Training & Evaluation (Weeks 3-4)

- Train TCN models on the same datasets used for RNN training
- Compare performance metrics between TCN and RNN approaches
- Perform ablation studies to identify optimal hyperparameters:
  - Number of filters
  - Kernel size
  - Dilation pattern
  - Receptive field size

### 4.3 Phase 3: Mobile Optimization (Weeks 5-6)

- Convert models to TensorFlow Lite and Core ML
- Benchmark inference time on mobile devices
- Optimize for reduced model size and power consumption
- Test integration with the real-time IMU data pipeline

### 4.4 Phase 4: Testing & Deployment (Weeks 7-8)

- Conduct real-world testing with controlled GPS outages
- Compare TCN accuracy with existing RNN approach
- Prepare integration with the main application
- Document findings and prepare technical report

## 5. Coding Implementation Details

### 5.1 TCN Block Implementation

```python
def tcn_block(inputs, filters, kernel_size, dilation_rate, dropout_rate=0.1):
    """
    Implements a single TCN block with residual connection
    
    Args:
        inputs: Input tensor
        filters: Number of convolutional filters
        kernel_size: Size of the convolutional kernel
        dilation_rate: Dilation rate (for exponential receptive field growth)
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Output tensor after applying TCN block
    """
    # Compute padding to maintain sequence length
    padding = (kernel_size - 1) * dilation_rate
    
    # First causal convolution
    x = tf.keras.layers.ZeroPadding1D(padding=(padding, 0))(inputs)
    x = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding='valid',
        activation='linear',
        kernel_initializer='he_normal'
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Second causal convolution
    x = tf.keras.layers.ZeroPadding1D(padding=(padding, 0))(x)
    x = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding='valid',
        activation='linear',
        kernel_initializer='he_normal'
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Add residual connection if input and output shapes match
    if inputs.shape[-1] != filters:
        # Linear projection for dimension matching
        residual = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, padding='same')(inputs)
    else:
        residual = inputs
        
    return tf.keras.layers.add([x, residual])
```

### 5.2 Complete TCN Model

```python
def create_tcn_dead_reckoning_model(sequence_length, num_features):
    """
    Create a TCN model for dead reckoning during GPS outages.
    
    Args:
        sequence_length: Length of input sequences (time steps)
        num_features: Number of IMU channels (typically 6 for accel+gyro)
    
    Returns:
        A compiled Keras model
    """
    # Input layers
    imu_input = tf.keras.layers.Input(shape=(sequence_length, num_features), name="imu_input")
    initial_state = tf.keras.layers.Input(shape=(6,), name="initial_state")  # [vx, vy, vz, px, py, pz]
    
    # TCN layers with increasing dilation rates
    x = imu_input
    n_filters = 64
    
    # Create TCN blocks with dilated convolutions
    dilation_rates = [1, 2, 4, 8, 16, 32]
    for dilation_rate in dilation_rates:
        x = tcn_block(
            x, 
            filters=n_filters, 
            kernel_size=3, 
            dilation_rate=dilation_rate, 
            dropout_rate=0.2
        )
    
    # Global pooling to reduce temporal dimension
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Combine with initial state
    combined = tf.keras.layers.Concatenate()([x, initial_state])
    
    # Fully connected layers
    x = tf.keras.layers.Dense(128, activation='relu')(combined)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Output layers - predict position and velocity deltas
    position_delta = tf.keras.layers.Dense(3, name="position_delta")(x)  # dx, dy, dz
    velocity_delta = tf.keras.layers.Dense(3, name="velocity_delta")(x)  # dvx, dvy, dvz
    
    # Create model
    model = tf.keras.Model(
        inputs=[imu_input, initial_state], 
        outputs=[position_delta, velocity_delta]
    )
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            "position_delta": "mse",
            "velocity_delta": "mse"
        },
        loss_weights={
            "position_delta": 1.0,
            "velocity_delta": 0.5
        },
        metrics={
            "position_delta": ["mae", "mse"],
            "velocity_delta": ["mae", "mse"]
        }
    )
    
    return model
```

### 5.3 PyTorch Implementation

```python
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout = dropout
        
        # First causal convolution
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, 
            padding=0, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # Second causal convolution
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, 
            padding=0, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Residual connection
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)
            
        # Dropout for regularization
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x):
        # Add padding for causal convolution
        padding = (self.kernel_size - 1) * self.dilation
        
        # First convolution
        out = F.pad(x, (padding, 0))
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout_layer(out)
        
        # Second convolution
        out = F.pad(out, (padding, 0))
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout_layer(out)
        
        # Residual connection
        res = x
        if self.downsample is not None:
            res = self.downsample(x)
            
        return F.relu(out + res)

class TCNDeadReckoning(nn.Module):
    def __init__(self, input_channels=6, n_filters=64, kernel_size=3, dropout=0.2):
        super(TCNDeadReckoning, self).__init__()
        
        # Define TCN with dilated convolutions
        self.tcn_layers = nn.ModuleList()
        
        # First layer with input channels
        self.tcn_layers.append(
            TCNBlock(input_channels, n_filters, kernel_size, dilation=1, dropout=dropout)
        )
        
        # Subsequent layers with increasing dilation rates
        dilation_rates = [2, 4, 8, 16, 32]
        for dilation in dilation_rates:
            self.tcn_layers.append(
                TCNBlock(n_filters, n_filters, kernel_size, dilation=dilation, dropout=dropout)
            )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers after combining with initial state
        self.fc1 = nn.Linear(n_filters + 6, 128)  # +6 for the initial state
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout)
        
        # Output heads
        self.position_delta = nn.Linear(64, 3)
        self.velocity_delta = nn.Linear(64, 3)
        
    def forward(self, imu_input, initial_state):
        # Transpose input for Conv1D (batch, channels, sequence_length)
        x = imu_input.transpose(1, 2)
        
        # Apply TCN layers
        for layer in self.tcn_layers:
            x = layer(x)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Combine with initial state
        x = torch.cat([x, initial_state], dim=1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        # Output heads
        position_delta = self.position_delta(x)
        velocity_delta = self.velocity_delta(x)
        
        return position_delta, velocity_delta
```

## 6. Evaluation Metrics

We will evaluate the TCN model using the following metrics:

1. **Position Accuracy**:
   - Mean Absolute Error (MAE)
   - Root Mean Square Error (RMSE)
   - Positional drift rate (m/s)

2. **Computational Efficiency**:
   - Inference time (ms)
   - Memory usage (MB)
   - Power consumption

3. **Outage Duration Performance**:
   - Accuracy by outage duration (10s, 30s, 60s)
   - Error growth rate analysis

## 7. Comparison with Current RNN Approach

| Metric | RNN/LSTM | TCN | Improvement |
|--------|----------|-----|-------------|
| Position RMSE (m) | [Baseline] | [Expected] | [Target %] |
| Inference Time (ms) | [Baseline] | [Expected] | [Target %] |
| Model Size (MB) | [Baseline] | [Expected] | [Target %] |
| Battery Impact | [Baseline] | [Expected] | [Target %] |
| Long-term Drift | [Baseline] | [Expected] | [Target %] |

## 8. Implementation Timeline

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 1 | - Design TCN architecture<br>- Implement core TCN blocks | - TCN block implementations<br>- Architecture diagram |
| 2 | - Complete model implementation<br>- Set up data pipeline<br>- Prepare training harness | - Complete TCN model code<br>- Data preprocessing pipeline |
| 3 | - Initial model training<br>- Hyperparameter experiments<br>- Compare with RNN baseline | - Trained prototype model<br>- Initial metrics comparison |
| 4 | - Fine-tune hyperparameters<br>- Advanced physics-based loss functions<br>- Ablation studies | - Optimized model<br>- Performance analysis report |
| 5 | - TensorFlow Lite conversion<br>- Mobile optimization<br>- Benchmark on target devices | - TFLite and CoreML models<br>- Mobile benchmarks |
| 6 | - Implement real-time processing<br>- Integration with IMU buffer<br>- Memory optimization | - Mobile integration code<br>- Performance analytics |
| 7 | - Real-world testing<br>- GPS outage simulations<br>- Challenging environment testing | - Field test results<br>- Comparison metrics |
| 8 | - Final optimizations<br>- Documentation<br>- Project wrap-up | - Final model and code<br>- Technical report<br>- Integration guide |

## 9. Integration Plan

To integrate the TCN model into our existing Vehicle Motion Tracking system:

1. **Android Integration**:
   - Create TCNDeadReckoning class in Kotlin
   - Configure TensorFlow Lite integration
   - Connect to IMU data stream
   - Implement buffer management for processing

2. **iOS Integration**:
   - Implement CoreML wrapper
   - Optimize for Apple Neural Engine
   - Match Android implementation for consistency

3. **Shared UI**:
   - Update visualization to show TCN results vs RNN
   - Add model selection toggle
   - Create error analysis dashboard

## 10. Risks and Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|------------|------------|
| TCN model too large for mobile | High | Medium | Quantization, pruning, architecture optimization |
| Accuracy degradation vs RNN | High | Low | Hybrid model approach, careful hyperparameter tuning |
| Training instability | Medium | Low | Gradual dilation increases, batch normalization |
| Excessive battery consumption | High | Low | Model optimization, efficient inference scheduling |
| Integration complexity | Medium | Medium | Incremental integration approach, detailed testing plan |

## 11. Resources Required

1. **Development Environment**:
   - TensorFlow 2.x / PyTorch 1.x
   - Python data science stack (NumPy, Pandas, Matplotlib)
   - Mobile development tools (Android Studio, Xcode)

2. **Hardware**:
   - GPU-enabled development machine
   - Test devices (Android and iOS)
   - IMU test rig for controlled experiments

3. **Data**:
   - Existing IMU datasets
   - Synthetic generated motion data
   - Real-world driving/walking datasets

## 12. Next Steps

1. Set up project repository and development environment
2. Implement core TCN architecture and test with synthetic data
3. Prepare training pipeline and evaluation metrics
4. Begin initial training and comparative analysis

This project plan provides a comprehensive roadmap for implementing TCN-based dead reckoning to enhance our position tracking capabilities during GPS outages.
