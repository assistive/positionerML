"""
Train an RNN-based model for IMU data smoothing and export it to TensorFlow Lite
for use in an Android Kotlin application.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input

def create_imu_smoothing_model(sequence_length, num_features):
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

def prepare_sequences(data, sequence_length):
    """
    Convert IMU data into overlapping sequences for training
    
    Args:
        data: Array of IMU readings [n_samples, n_features]
        sequence_length: Length of sequences to create
        
    Returns:
        Array of overlapping sequences
    """
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i+sequence_length])
    return np.array(sequences)

def add_synthetic_noise(clean_data, noise_level=0.05):
    """
    Add synthetic noise to clean data to create noisy training examples
    
    Args:
        clean_data: Clean IMU data
        noise_level: Standard deviation of Gaussian noise to add
        
    Returns:
        Noisy version of the data
    """
    noise = np.random.normal(0, noise_level, clean_data.shape)
    noisy_data = clean_data + noise
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

def plot_comparison(raw_data, smoothed_data, start_idx=0, length=200, feature_idx=0):
    """
    Plot comparison between raw and smoothed data
    
    Args:
        raw_data: Raw IMU data
        smoothed_data: RNN-smoothed data
        start_idx: Starting index for plot
        length: Number of samples to plot
        feature_idx: Which IMU feature to plot (0-5 for accel/gyro)
    """
    end_idx = start_idx + length
    
    plt.figure(figsize=(12, 6))
    plt.plot(raw_data[start_idx:end_idx, feature_idx], label='Raw data', alpha=0.7)
    plt.plot(smoothed_data[start_idx:end_idx, feature_idx], label='Smoothed data', linewidth=2)
    plt.legend()
    plt.title(f'Comparison of Raw vs. Smoothed IMU Data (Feature {feature_idx})')
    plt.xlabel('Time steps')
    plt.ylabel('Acceleration/Angular velocity')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'imu_smoothing_comparison_feature_{feature_idx}.png')
    plt.show()

def load_and_preprocess_data(file_path):
    """
    Load IMU data from CSV file
    
    Args:
        file_path: Path to CSV file with IMU data
        
    Returns:
        Numpy array with IMU data
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Select relevant columns - adjust these to match your actual data format
    imu_columns = [
        'accel_x', 'accel_y', 'accel_z',
        'gyro_x', 'gyro_y', 'gyro_z'
    ]
    
    # Extract IMU data
    imu_data = df[imu_columns].values
    
    # Normalize data
    mean = np.mean(imu_data, axis=0)
    std = np.std(imu_data, axis=0)
    normalized_data = (imu_data - mean) / std
    
    return normalized_data, mean, std

def main():
    # Parameters
    sequence_length = 50  # Length of sequences for RNN
    num_features = 6      # 3 accel + 3 gyro
    epochs = 30
    batch_size = 32
    
    # Get IMU data - replace with your data path
    data_file = "imu_data.csv"  # Replace with your data file
    
    # If you have real data:
    if os.path.exists(data_file):
        print(f"Loading data from {data_file}")
        imu_data, mean, std = load_and_preprocess_data(data_file)
    else:
        # Generate synthetic data for demonstration
        print("Generating synthetic data")
        num_samples = 10000
        t = np.linspace(0, 100, num_samples)
        
        # Create synthetic IMU data with patterns similar to vehicle motion
        accel_x = 0.1 * np.sin(0.1 * t) + 0.05 * np.sin(0.5 * t)
        accel_y = 0.15 * np.sin(0.08 * t + 1) + 0.1 * np.sin(0.4 * t)
        accel_z = 0.9 + 0.05 * np.sin(0.2 * t + 2) + 0.02 * np.sin(0.7 * t)
        
        gyro_x = 0.02 * np.sin(0.3 * t + 1) + 0.01 * np.sin(0.9 * t)
        gyro_y = 0.03 * np.sin(0.25 * t + 2) + 0.015 * np.sin(0.8 * t)
        gyro_z = 0.01 * np.sin(0.4 * t) + 0.02 * np.sin(0.6 * t + 3)
        
        # Combine into single array
        imu_data = np.column_stack((accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z))
        
        # Add sudden events (like braking)
        for i in range(5):
            start_idx = np.random.randint(100, num_samples - 200)
            # Simulate braking
            imu_data[start_idx:start_idx+30, 0] -= np.linspace(0, 0.5, 30)
            
        # Add turning events
        for i in range(5):
            start_idx = np.random.randint(100, num_samples - 200)
            # Simulate turning
            imu_data[start_idx:start_idx+50, 1] += np.linspace(0, 0.4, 50)
            imu_data[start_idx:start_idx+50, 5] += np.linspace(0, 0.1, 50)
        
        # Normalization parameters (for denormalization later)
        mean = np.mean(imu_data, axis=0)
        std = np.std(imu_data, axis=0)
    
    # Create "clean" target data using low-pass filtering
    clean_data = apply_low_pass_filter(imu_data, alpha=0.1)
    
    # Add more noise to create noisy training data
    noisy_data = add_synthetic_noise(imu_data, noise_level=0.1)
    
    # Prepare sequences
    X = prepare_sequences(noisy_data, sequence_length)
    y = prepare_sequences(clean_data, sequence_length)
    
    # Train-validation-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Create model
    model = create_imu_smoothing_model(sequence_length, num_features)
    model.summary()
    
    # Train with early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'imu_smoother_best.h5', monitor='val_loss', save_best_only=True
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint]
    )
    
    # Evaluate on test set
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")
    
    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    # Test on a continuous section of data
    test_idx = np.random.randint(0, len(imu_data) - 1000)
    test_data = noisy_data[test_idx:test_idx+1000]
    
    # Apply model to test data
    smoothed_sequences = []
    for i in range(len(test_data) - sequence_length + 1):
        seq = test_data[i:i+sequence_length]
        smoothed_seq = model.predict(np.expand_dims(seq, axis=0))[0]
        smoothed_sequences.append(smoothed_seq[-1])  # Take last prediction
    
    # Add padding at the beginning (since we can't predict the first sequence_length-1 points)
    padding = np.zeros((sequence_length-1, num_features))
    smoothed_data = np.vstack([padding, np.array(smoothed_sequences)])
    
    # Plot comparison
    for feature_idx in range(num_features):
        plot_comparison(test_data, smoothed_data, start_idx=sequence_length, 
                        length=500, feature_idx=feature_idx)
    
    # Export model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save TFLite model
    with open('imu_smoother.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("Model saved as 'imu_smoother.tflite'")
    
    # Save normalization parameters for use in the app
    np.savez('normalization_params.npz', mean=mean, std=std)
    print("Normalization parameters saved to 'normalization_params.npz'")

if __name__ == "__main__":
    main()
