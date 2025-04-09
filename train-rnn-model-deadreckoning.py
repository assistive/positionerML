"""
Enhanced RNN training script specifically optimized for dead reckoning during GPS outages.
This script trains a model to predict position changes from IMU data, allowing for
accurate position estimation for up to 60 seconds without GPS.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import argparse

def create_dead_reckoning_model(sequence_length, num_features):
    """
    Create a deep RNN model for dead reckoning during GPS outages.
    The model takes IMU sequences and predicts position deltas.
    
    Args:
        sequence_length: Length of input sequences (time steps)
        num_features: Number of IMU channels (typically 6 for accel+gyro)
    
    Returns:
        A compiled Keras model
    """
    # Input layer
    imu_input = Input(shape=(sequence_length, num_features), name="imu_input")
    
    # Initial velocity and position input (optional)
    initial_state = Input(shape=(6,), name="initial_state")  # [vx, vy, vz, px, py, pz]
    
    # Bidirectional LSTM layers for IMU processing
    x = Bidirectional(LSTM(128, return_sequences=True))(imu_input)
    x = Dropout(0.2)(x)
    
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = Dropout(0.2)(x)
    
    # Combine with initial state
    combined = Concatenate()([x, initial_state])
    
    # Fully connected layers
    x = Dense(128, activation='relu')(combined)
    x = Dropout(0.2)(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Output layers - predict position delta
    position_delta = Dense(3, name="position_delta")(x)  # dx, dy, dz
    velocity_delta = Dense(3, name="velocity_delta")(x)  # dvx, dvy, dvz
    
    # Create model
    model = Model(inputs=[imu_input, initial_state], 
                  outputs=[position_delta, velocity_delta])
    
    # Compile model with MSE loss and custom metrics
    model.compile(
        optimizer=Adam(learning_rate=0.001),
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

def create_smoothing_model(sequence_length, num_features):
    """
    Create an RNN model for IMU data smoothing
    
    Args:
        sequence_length: Length of input sequences (time steps)
        num_features: Number of IMU channels (typically 6 for accel+gyro)
    
    Returns:
        A compiled Keras model
    """
    model = Sequential([
        # Input layer
        Input(shape=(sequence_length, num_features)),
        
        # Bidirectional LSTM layers
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        
        Bidirectional(LSTM(32, return_sequences=True)),
        Dropout(0.2),
        
        # Output layer - same shape as input for sequence-to-sequence
        Dense(num_features)
    ])
    
    # Compile model with MSE loss
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

def prepare_sequences(imu_data, position_data, sequence_length, step=1):
    """
    Prepare training data with IMU sequences and corresponding position changes
    
    Args:
        imu_data: Array of IMU readings [n_samples, n_features]
        position_data: Array of position data [n_samples, 3]
        sequence_length: Length of sequences to create
        step: Step size between sequences (default=1)
        
    Returns:
        X_imu: IMU sequences
        X_initial: Initial position/velocity
        Y_pos_delta: Position changes
        Y_vel_delta: Velocity changes
    """
    X_imu = []
    X_initial = []
    Y_pos_delta = []
    Y_vel_delta = []
    
    # Estimate velocities from position changes (simple finite difference)
    velocities = np.zeros_like(position_data)
    for i in range(1, len(position_data)):
        velocities[i] = (position_data[i] - position_data[i-1]) / 0.01  # Assuming 100Hz (0.01s)
    
    # Create sequences
    for i in range(0, len(imu_data) - sequence_length - 1, step):
        # IMU sequence
        X_imu.append(imu_data[i:i+sequence_length])
        
        # Initial state (position and velocity at start of sequence)
        X_initial.append(np.concatenate([velocities[i], position_data[i]]))
        
        # Target: position and velocity change from start to end of sequence
        Y_pos_delta.append(position_data[i+sequence_length] - position_data[i])
        Y_vel_delta.append(velocities[i+sequence_length] - velocities[i])
    
    return np.array(X_imu), np.array(X_initial), np.array(Y_pos_delta), np.array(Y_vel_delta)

def add_synthetic_noise(data, noise_level=0.05):
    """
    Add synthetic noise to clean data to create noisy training examples
    
    Args:
        data: Clean IMU data
        noise_level: Standard deviation of Gaussian noise to add
        
    Returns:
        Noisy version of the data
    """
    noise = np.random.normal(0, noise_level, data.shape)
    noisy_data = data + noise
    return noisy_data

def apply_low_pass_filter(data, alpha=0.2):
    """
    Apply a simple low-pass filter to create "clean" target data
    
    Args:
        data: Raw IMU data
        alpha: Filter parameter (0-1), lower values = more smoothing
        
    Returns:
        Filtered data
    """
    filtered_data = np.zeros_like(data)
    filtered_data[0] = data[0]
    
    for i in range(1, len(data)):
        filtered_data[i] = alpha * data[i] + (1 - alpha) * filtered_data[i-1]
    
    return filtered_data

def generate_synthetic_trajectory(duration_seconds, sampling_rate=100):
    """
    Generate a synthetic vehicle trajectory with IMU and position data
    
    Args:
        duration_seconds: Length of trajectory in seconds
        sampling_rate: Samples per second
    
    Returns:
        imu_data: Array of synthetic IMU readings [n_samples, 6]
        position_data: Array of synthetic positions [n_samples, 3]
    """
    num_samples = int(duration_seconds * sampling_rate)
    t = np.linspace(0, duration_seconds, num_samples)
    
    # Generate position trajectory
    x = np.zeros(num_samples)
    y = np.zeros(num_samples)
    z = np.zeros(num_samples)
    
    # Current velocity
    vx = 0.0
    vy = 0.0
    vz = 0.0
    
    # Acceleration noise
    ax_noise = np.random.normal(0, 0.1, num_samples)
    ay_noise = np.random.normal(0, 0.1, num_samples)
    az_noise = np.random.normal(0, 0.2, num_samples)
    
    # Gyroscope noise (angular velocity)
    gx_noise = np.random.normal(0, 0.05, num_samples)
    gy_noise = np.random.normal(0, 0.05, num_samples)
    gz_noise = np.random.normal(0, 0.05, num_samples)
    
    # Generate motion patterns
    for i in range(1, num_samples):
        dt = 1.0 / sampling_rate
        
        # Decide if we generate a motion event
        if np.random.random() < 0.005:  # 0.5% chance of starting an event
            event_type = np.random.choice(['acceleration', 'braking', 'turning'])
            event_duration = np.random.randint(1, 3) * sampling_rate  # 1-3 seconds
            
            if event_type == 'acceleration':
                ax = 2.0 + np.random.random()  # 2-3 m/s²
                ay = 0.0
                az = 0.0
            elif event_type == 'braking':
                ax = -3.0 - np.random.random()  # -3 to -4 m/s²
                ay = 0.0
                az = 0.0
            elif event_type == 'turning':
                ax = 0.0
                ay = 2.0 if np.random.random() < 0.5 else -2.0  # Left or right turn
                az = 0.0
                # Add angular velocity for turning
                gz_noise[i:i+event_duration] += 0.2 * (1 if ay > 0 else -1)
                
            # Apply the acceleration for the event duration
            for j in range(i, min(i + event_duration, num_samples)):
                # Update velocity
                vx += ax * dt
                vy += ay * dt
                vz += az * dt
                
                # Update position
                x[j] = x[j-1] + vx * dt
                y[j] = y[j-1] + vy * dt
                z[j] = z[j-1] + vz * dt
                
                # Add noise to acceleration
                ax_noise[j] += ax
                ay_noise[j] += ay
                az_noise[j] += az
            
            # Skip to end of event
            i += event_duration
        else:
            # Normal driving with slight changes
            ax = ax_noise[i]
            ay = ay_noise[i]
            az = az_noise[i]
            
            # Update velocity with natural damping (friction)
            vx = vx * 0.99 + ax * dt
            vy = vy * 0.99 + ay * dt
            vz = vz * 0.99 + az * dt
            
            # Update position
            x[i] = x[i-1] + vx * dt
            y[i] = y[i-1] + vy * dt
            z[i] = z[i-1] + vz * dt
    
    # Combine into arrays
    position_data = np.column_stack((x, y, z))
    
    # Calculate true accelerations from position (second derivative)
    accel_x = np.zeros(num_samples)
    accel_y = np.zeros(num_samples)
    accel_z = np.zeros(num_samples)
    
    for i in range(1, num_samples-1):
        accel_x[i] = (x[i+1] - 2*x[i] + x[i-1]) / (dt*dt)
        accel_y[i] = (y[i+1] - 2*y[i] + y[i-1]) / (dt*dt)
        accel_z[i] = (z[i+1] - 2*z[i] + z[i-1]) / (dt*dt)
    
    # Fix endpoints
    accel_x[0] = accel_x[1]
    accel_y[0] = accel_y[1]
    accel_z[0] = accel_z[1]
    accel_x[-1] = accel_x[-2]
    accel_y[-1] = accel_y[-2]
    accel_z[-1] = accel_z[-2]
    
    # Create IMU data by adding noise to true accelerations
    imu_data = np.column_stack((
        accel_x + ax_noise,
        accel_y + ay_noise,
        accel_z + az_noise,
        gx_noise,
        gy_noise,
        gz_noise
    ))
    
    return imu_data, position_data

def simulate_gps_outages(position_data, outage_count=10, max_outage_length=600):
    """
    Simulate GPS outages by removing segments of position data
    
    Args:
        position_data: Array of position data [n_samples, 3]
        outage_count: Number of outages to simulate
        max_outage_length: Maximum length of outage in samples
        
    Returns:
        masked_positions: Position data with outages (NaN values)
        outage_mask: Boolean mask where True indicates available data
    """
    masked_positions = position_data.copy()
    outage_mask = np.ones(len(position_data), dtype=bool)
    
    for _ in range(outage_count):
        # Choose random outage length
        outage_length = np.random.randint(50, max_outage_length)
        
        # Choose random start point (avoiding overlaps)
        valid_starts = np.where(outage_mask)[0]
        if len(valid_starts) < outage_length:
            break
            
        # Find a segment long enough for the outage
        valid_segments = []
        current_segment = []
        
        for i in range(1, len(valid_starts)):
            if valid_starts[i] == valid_starts[i-1] + 1:
                current_segment.append(valid_starts[i-1])
                if i == len(valid_starts) - 1:
                    current_segment.append(valid_starts[i])
                    valid_segments.append(current_segment)
            else:
                if current_segment:
                    current_segment.append(valid_starts[i-1])
                    valid_segments.append(current_segment)
                current_segment = []
                    
        # Filter segments that are long enough
        valid_segments = [seg for seg in valid_segments if len(seg) >= outage_length]
        
        if not valid_segments:
            break
            
        # Choose a random segment
        chosen_segment = valid_segments[np.random.randint(0, len(valid_segments))]
        
        # Choose random start within segment
        max_start = len(chosen_segment) - outage_length
        start_idx = np.random.randint(0, max_start)
        outage_indices = chosen_segment[start_idx:start_idx + outage_length]
        
        # Create the outage
        outage_mask[outage_indices] = False
        masked_positions[outage_indices] = np.nan
    
    return masked_positions, outage_mask

def plot_trajectory_with_outages(true_positions, masked_positions, predicted_positions=None):
    """
    Plot the trajectory showing true path, GPS with outages, and predictions
    
    Args:
        true_positions: Ground truth position data [n_samples, 3]
        masked_positions: Position data with outages (NaN values)
        predicted_positions: Predicted positions during outages (optional)
    """
    plt.figure(figsize=(12, 10))
    
    # Create 3D plot
    ax = plt.subplot(111, projection='3d')
    
    # Plot true trajectory
    ax.plot(true_positions[:, 0], true_positions[:, 1], true_positions[:, 2], 
            color='green', label='Ground Truth', linewidth=2)
    
    # Plot available GPS points
    valid_mask = ~np.isnan(masked_positions[:, 0])
    ax.scatter(masked_positions[valid_mask, 0], 
              masked_positions[valid_mask, 1], 
              masked_positions[valid_mask, 2],
              color='blue', label='Available GPS', s=10)
    
    # Plot predicted trajectory if provided
    if predicted_positions is not None:
        # Find outage segments
        outage_start = None
        for i in range(1, len(valid_mask)):
            if valid_mask[i-1] and not valid_mask[i]:  # Start of outage
                outage_start = i
            elif not valid_mask[i-1] and valid_mask[i]:  # End of outage
                if outage_start is not None:
                    # Plot this outage segment
                    segment = predicted_positions[outage_start:i]
                    ax.plot(segment[:, 0], segment[:, 1], segment[:, 2],
                            color='red', linewidth=2)
                    outage_start = None
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('Vehicle Trajectory with GPS Outages and Dead Reckoning')
    ax.legend()
    
    plt.savefig('trajectory_with_outages.png')
    plt.close()

def dead_reckon(imu_data, initial_state, model, sequence_length, dt=0.01):
    """
    Perform dead reckoning using the trained model
    
    Args:
        imu_data: IMU data array [n_samples, 6]
        initial_state: Initial position and velocity [vx, vy, vz, px, py, pz]
        model: Trained dead reckoning model
        sequence_length: Length of sequences used for model
        dt: Time step between samples
        
    Returns:
        Estimated position trajectory [n_samples, 3]
    """
    # Initialize result arrays
    n_samples = len(imu_data)
    positions = np.zeros((n_samples, 3))
    velocities = np.zeros((n_samples, 3))
    
    # Set initial values
    positions[0] = initial_state[3:]
    velocities[0] = initial_state[:3]
    
    # Current state
    current_pos = positions[0].copy()
    current_vel = velocities[0].copy()
    current_state = np.concatenate([current_vel, current_pos])
    
    # Process data in overlapping windows
    step_size = max(1, sequence_length // 2)  # 50% overlap
    
    for i in range(0, n_samples - sequence_length, step_size):
        # Extract sequence
        imu_sequence = imu_data[i:i+sequence_length]
        imu_sequence = imu_sequence.reshape(1, sequence_length, 6)  # Add batch dimension
        
        # Prepare current state
        current_state_input = current_state.reshape(1, 6)
        
        # Predict position and velocity deltas
        pos_delta, vel_delta = model.predict([imu_sequence, current_state_input], verbose=0)
        
        # Update position and velocity
        current_vel = current_vel + vel_delta[0]
        current_pos = current_pos + pos_delta[0]
        
        # Update state for next prediction
        current_state = np.concatenate([current_vel, current_pos])
        
        # Store values for each step in the window
        for j in range(step_size):
            if i + j < n_samples:
                positions[i + j] = current_pos
                velocities[i + j] = current_vel
    
    # Fill any remaining positions
    if n_samples > i + step_size:
        for j in range(i + step_size, n_samples):
            positions[j] = current_pos
            velocities[j] = current_vel
    
    return positions

def evaluate_dead_reckoning(model, test_imu, test_positions, sequence_length):
    """
    Evaluate dead reckoning performance on test data
    
    Args:
        model: Trained dead reckoning model
        test_imu: Test IMU data
        test_positions: Ground truth positions
        sequence_length: Sequence length for model
        
    Returns:
        mean_error: Mean position error in meters
        max_error: Maximum position error in meters
    """
    # Simulate GPS outages
    masked_positions, outage_mask = simulate_gps_outages(
        test_positions, outage_count=5, max_outage_length=600)
    
    # Initialize predicted positions with the masked positions
    predicted_positions = masked_positions.copy()
    
    # Find outage segments
    outage_segments = []
    outage_start = None
    
    for i in range(1, len(outage_mask)):
        if outage_mask[i-1] and not outage_mask[i]:  # Start of outage
            outage_start = i
        elif not outage_mask[i-1] and outage_mask[i]:  # End of outage
            if outage_start is not None:
                outage_segments.append((outage_start, i))
                outage_start = None
    
    # Process each outage
    errors = []
    
    for start, end in outage_segments:
        # Get initial state (from last known position before outage)
        if start > 0:
            last_known_pos = test_positions[start-1]
            
            # Estimate velocity from previous positions
            if start > 5:
                last_velocity = (test_positions[start-1] - test_positions[start-6]) / (5 * 0.01)
            else:
                last_velocity = np.zeros(3)
                
            initial_state = np.concatenate([last_velocity, last_known_pos])
            
            # Perform dead reckoning through the outage
            outage_length = end - start
            outage_imu = test_imu[start:end]
            
            # Ensure we have enough data for at least one sequence
            if len(outage_imu) >= sequence_length:
                dr_positions = dead_reckon(
                    outage_imu, initial_state, model, sequence_length)
                
                # Fill in predicted positions
                predicted_positions[start:end] = dr_positions[:outage_length]
                
                # Calculate errors
                segment_errors = np.linalg.norm(
                    predicted_positions[start:end] - test_positions[start:end], axis=1)
                errors.extend(segment_errors)
    
    # Calculate error statistics
    mean_error = np.mean(errors) if errors else 0
    max_error = np.max(errors) if errors else 0
    
    # Plot results
    plot_trajectory_with_outages(test_positions, masked_positions, predicted_positions)
    
    return mean_error, max_error

def main():
    parser = argparse.ArgumentParser(description='Train RNN models for IMU data processing')
    parser.add_argument('--data', type=str, help='Path to input data CSV')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--sequence-length', type=int, default=50, help='Sequence length for RNN')
    parser.add_argument('--synthetic', action='store_true', help='Generate synthetic data')
    parser.add_argument('--mode', type=str, default='both', 
                       choices=['smoothing', 'dead_reckoning', 'both'],
                       help='Which model(s) to train')
    
    args = parser.parse_args()
    
    # Parameters
    sequence_length = args.sequence_length
    num_features = 6      # 3 accel + 3 gyro
    epochs = args.epochs
    batch_size = args.batch_size
    
    # Load or generate data
    if args.synthetic or not args.data or not os.path.exists(args.data):
        print("Generating synthetic data...")
        # Generate 5 minutes of data at 100Hz
        imu_data, position_data = generate_synthetic_trajectory(300, sampling_rate=100)
        
        # Add more noise to create noisy training data
        noisy_imu_data = add_synthetic_noise(imu_data, noise_level=0.2)
    else:
        print(f"Loading data from {args.data}")
        # Load your CSV data here
        # This is a placeholder - adapt to your data format
        df = pd.read_csv(args.data)
        
        # Extract IMU and position data
        imu_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
        pos_cols = ['position_x', 'position_y', 'position_z']
        
        imu_data = df[imu_cols].values
        position_data = df[pos_cols].values
        
        # Normalize data
        imu_mean = np.mean(imu_data, axis=0)
        imu_std = np.std(imu_data, axis=0)
        imu_data = (imu_data - imu_mean) / imu_std
        
        # Create noisy version
        noisy_imu_data = add_synthetic_noise(imu_data, noise_level=0.1)
    
    # Create "clean" target data using low-pass filtering for smoothing model
    clean_imu_data = apply_low_pass_filter(imu_data, alpha=0.1)
    
    # Split data into train/validation/test sets
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    # Train/test split
    train_idx = int(len(imu_data) * train_ratio)
    val_idx = train_idx + int(len(imu_data) * val_ratio)
    
    # Split data
    train_imu = noisy_imu_data[:train_idx]
    val_imu = noisy_imu_data[train_idx:val_idx]
    test_imu = noisy_imu_data[val_idx:]
    
    train_clean_imu = clean_imu_data[:train_idx]
    val_clean_imu = clean_imu_data[train_idx:val_idx]
    
    train_pos = position_data[:train_idx]
    val_pos = position_data[train_idx:val_idx]
    test_pos = position_data[val_idx:]
    
    # PART 1: Train IMU Smoothing Model
    if args.mode in ['smoothing', 'both']:
        print("Training IMU Smoothing Model...")
        
        # Prepare sequences for smoothing
        X_train = []
        y_train = []
        
        for i in range(len(train_imu) - sequence_length):
            X_train.append(train_imu[i:i+sequence_length])
            y_train.append(train_clean_imu[i:i+sequence_length])
            
        X_val = []
        y_val = []
        
        for i in range(len(val_imu) - sequence_length):
            X_val.append(val_imu[i:i+sequence_length])
            y_val.append(val_clean_imu[i:i+sequence_length])
            
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        
        # Create and train the smoothing model
        smoothing_model = create_smoothing_model(sequence_length, num_features)
        
        # Callbacks for training
        smoothing_callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ModelCheckpoint('imu_smoother_best.h5', save_best_only=True),
            ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5)
        ]
        
        smoothing_history = smoothing_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=smoothing_callbacks
        )
        
        # Save smoothing model to TensorFlow Lite format
        converter = tf.lite.TFLiteConverter.from_keras_model(smoothing_model)
        tflite_model = converter.convert()
        
        with open('imu_smoother.tflite', 'wb') as f:
            f.write(tflite_model)
            
        print("Smoothing model saved as 'imu_smoother.tflite'")
        
    # PART 2: Train Dead Reckoning Model
    if args.mode in ['dead_reckoning', 'both']:
        print("Training Dead Reckoning Model...")
        
        # Prepare sequences for dead reckoning
        X_imu_train, X_initial_train, y_pos_delta_train, y_vel_delta_train = prepare_sequences(
            train_imu, train_pos, sequence_length)
        
        X_imu_val, X_initial_val, y_pos_delta_val, y_vel_delta_val = prepare_sequences(
            val_imu, val_pos, sequence_length)
        
        # Create and train the dead reckoning model
        dead_reckoning_model = create_dead_reckoning_model(sequence_length, num_features)
        
        # Callbacks for training
        dr_callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint('dead_reckoning_best.h5', save_best_only=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5)
        ]
        
        dr_history = dead_reckoning_model.fit(
            [X_imu_train, X_initial_train], 
            [y_pos_delta_train, y_vel_delta_train],
            validation_data=(
                [X_imu_val, X_initial_val],
                [y_pos_delta_val, y_vel_delta_val]
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=dr_callbacks
        )
        
        # Save dead reckoning model to TensorFlow Lite format
        converter = tf.lite.TFLiteConverter.from_keras_model(dead_reckoning_model)
        tflite_model = converter.convert()
        
        with open('dead_reckoning.tflite', 'wb') as f:
            f.write(tflite_model)
            
        print("Dead reckoning model saved as 'dead_reckoning.tflite'")
        
        # Evaluate dead reckoning performance
        mean_error, max_error = evaluate_dead_reckoning(
            dead_reckoning_model, test_imu, test_pos, sequence_length)
        
        print(f"Dead reckoning evaluation:")
        print(f"  Mean position error: {mean_error:.2f} meters")
        print(f"  Maximum position error: {max_error:.2f} meters")
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(dr_history.history['position_delta_loss'], label='Position Loss')
        plt.plot(dr_history.history['val_position_delta_loss'], label='Val Position Loss')
        plt.title('Position Delta Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(dr_history.history['velocity_delta_loss'], label='Velocity Loss')
        plt.plot(dr_history.history['val_velocity_delta_loss'], label='Val Velocity Loss')
        plt.title('Velocity Delta Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('dead_reckoning_training.png')
        plt.close()
    
    print("Training complete!")

if __name__ == "__main__":
    main()
