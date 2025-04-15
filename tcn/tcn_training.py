#!/usr/bin/env python
# coding: utf-8

"""
Training script for the TCN IMU Smoother model.

This script trains a Temporal Convolutional Network (TCN) for IMU data smoothing.
The model expects a sequence of IMU data and outputs a smoothed version of the same data.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Activation, Add, Lambda, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_data(data_dir, file_pattern="*.csv"):
    """
    Load and preprocess IMU data from CSV files.
    
    Args:
        data_dir: Directory containing IMU data files
        file_pattern: Glob pattern to match data files
        
    Returns:
        X_train: Raw IMU sequences for training
        y_train: Filtered/smoothed ground truth for training
        X_val: Raw IMU sequences for validation
        y_val: Filtered/smoothed ground truth for validation
    """
    import glob
    
    # Find all matching data files
    data_files = glob.glob(os.path.join(data_dir, file_pattern))
    print(f"Found {len(data_files)} data files")
    
    raw_sequences = []
    truth_sequences = []
    
    for file_path in data_files:
        # Load data
        df = pd.read_csv(file_path)
        
        # Extract feature columns (accelerometer, gyroscope readings)
        # Adjust column names to match your data format
        feature_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
        
        if not all(col in df.columns for col in feature_cols):
            print(f"Warning: Missing expected columns in {file_path}")
            print(f"Available columns: {df.columns.tolist()}")
            continue
            
        # Extract raw IMU data
        raw_data = df[feature_cols].values
        
        # If you have ground truth smoothed data, use it
        # Otherwise, we'll generate synthetic smoothed data
        if 'smooth_accel_x' in df.columns:
            # Your data already contains ground truth
            smooth_cols = ['smooth_accel_x', 'smooth_accel_y', 'smooth_accel_z', 
                           'smooth_gyro_x', 'smooth_gyro_y', 'smooth_gyro_z']
            smooth_data = df[smooth_cols].values
        else:
            # Generate synthetic smoothed data using moving average
            window_size = 5
            smooth_data = np.zeros_like(raw_data)
            for i in range(raw_data.shape[1]):
                smooth_data[:, i] = pd.Series(raw_data[:, i]).rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
        
        # Create sequences of fixed length for training
        seq_length = 128  # Adjust based on your model architecture
        stride = 32       # Overlap between sequences
        
        for i in range(0, len(raw_data) - seq_length, stride):
            raw_seq = raw_data[i:i+seq_length]
            smooth_seq = smooth_data[i:i+seq_length]
            
            raw_sequences.append(raw_seq)
            truth_sequences.append(smooth_seq)
    
    # Convert to numpy arrays
    X = np.array(raw_sequences)
    y = np.array(truth_sequences)
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, y_train, X_val, y_val

def residual_block(x, filters, kernel_size, dilation_rate):
    """
    Create a residual block for the TCN.
    
    Args:
        x: Input tensor
        filters: Number of filters in the convolution
        kernel_size: Size of the convolution kernel
        dilation_rate: Dilation rate for dilated convolution
        
    Returns:
        Output tensor of the residual block
    """
    # Shortcut connection
    shortcut = x
    
    # Dilated causal convolution path
    conv1 = Conv1D(filters=filters,
                  kernel_size=kernel_size,
                  dilation_rate=dilation_rate,
                  padding='causal')(x)
    norm1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(norm1)
    
    conv2 = Conv1D(filters=filters,
                  kernel_size=kernel_size,
                  dilation_rate=dilation_rate,
                  padding='causal')(act1)
    norm2 = BatchNormalization()(conv2)
    act2 = Activation('relu')(norm2)
    
    # If input and output dimensions don't match, use a 1x1 convolution to match
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters=filters, kernel_size=1, padding='same')(shortcut)
    
    # Add the shortcut connection
    out = Add()([act2, shortcut])
    return out

def build_tcn_model(seq_length, n_features, n_filters=64):
    """
    Build a Temporal Convolutional Network (TCN) for IMU smoothing.
    
    Args:
        seq_length: Length of input sequences
        n_features: Number of features in each time step (6 for 3-axis accel + 3-axis gyro)
        n_filters: Number of filters in TCN blocks
        
    Returns:
        TCN model
    """
    # Input layer
    inputs = Input(shape=(seq_length, n_features))
    
    # Initial projection to n_filters
    x = Conv1D(filters=n_filters, kernel_size=1, padding='causal')(inputs)
    
    # TCN consists of multiple stacked dilated residual blocks
    # Dilation rates often follow exponential pattern: 1, 2, 4, 8, ...
    for dilation_rate in [1, 2, 4, 8, 16]:
        x = residual_block(x, filters=n_filters, kernel_size=3, dilation_rate=dilation_rate)
    
    # Final projection back to input dimension
    outputs = Conv1D(filters=n_features, kernel_size=1, padding='same', activation='linear')(x)
    
    # Create and compile model
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    return model

def train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Train the TCN model.
    
    Args:
        X_train: Training data
        y_train: Training targets
        X_val: Validation data
        y_val: Validation targets
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Trained model and training history
    """
    # Get shapes from data
    seq_length = X_train.shape[1]
    n_features = X_train.shape[2]
    
    # Build model
    model = build_tcn_model(seq_length, n_features)
    model.summary()
    
    # Setup callbacks
    checkpoint = ModelCheckpoint('tcn_model_best.h5', monitor='val_loss', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    return model, history

def plot_training_history(history):
    """
    Plot training history.
    
    Args:
        history: Training history from model.fit()
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def convert_to_tflite(model, output_path="tcn_imu_smoother.tflite"):
    """
    Convert the trained model to TensorFlow Lite format.
    
    Args:
        model: Trained Keras model
        output_path: Path to save the TFLite model
    """
    # Convert to TensorFlow Lite model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimize for mobile deployment
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert and save
    tflite_model = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Model saved to {output_path}")

def convert_to_coreml(model, output_path="TCNIMUSmoother.mlmodel"):
    """
    Convert the trained model to CoreML format for iOS.
    
    Args:
        model: Trained Keras model
        output_path: Path to save the CoreML model
    """
    # Try to import coremltools, which might not be installed
    try:
        import coremltools as ct
        
        # Convert to CoreML model
        mlmodel = ct.convert(model, 
                           inputs=[ct.TensorType(shape=(1, None, model.input_shape[-1]))],
                           minimum_deployment_target=ct.target.iOS14)
        
        # Save the model
        mlmodel.save(output_path)
        print(f"CoreML model saved to {output_path}")
        
    except ImportError:
        print("CoreMLTools not installed. To convert to CoreML format for iOS, install coremltools package.")
        print("pip install coremltools")

def visualize_model_outputs(model, X_val, y_val, num_samples=3):
    """
    Visualize some examples of model outputs vs ground truth.
    
    Args:
        model: Trained model
        X_val: Validation input data
        y_val: Validation target data
        num_samples: Number of samples to visualize
    """
    # Get predictions
    y_pred = model.predict(X_val[:num_samples])
    
    for i in range(num_samples):
        plt.figure(figsize=(15, 10))
        
        # Plot each feature channel separately
        feature_names = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
        
        for j, name in enumerate(feature_names):
            plt.subplot(3, 2, j+1)
            plt.plot(X_val[i, :, j], 'b-', alpha=0.5, label='Raw')
            plt.plot(y_val[i, :, j], 'g-', label='Ground Truth')
            plt.plot(y_pred[i, :, j], 'r-', label='Predicted')
            plt.title(f'{name}')
            plt.legend()
            
        plt.tight_layout()
        plt.savefig(f'sample_{i+1}_output.png')
        plt.close()

def main():
    # Set data directory
    data_dir = "imu_data"  # Change to your data directory
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, y_train, X_val, y_val = load_and_preprocess_data(data_dir)
    print(f"Data shapes: X_train={X_train.shape}, y_train={y_train.shape}")
    
    # Train model
    print("Training model...")
    model, history = train_model(X_train, y_train, X_val, y_val)
    
    # Plot training history
    plot_training_history(history)
    
    # Visualize model outputs
    print("Visualizing model outputs...")
    visualize_model_outputs(model, X_val, y_val)
    
    # Save the model in TensorFlow Lite format
    print("Converting model to TensorFlow Lite format...")
    convert_to_tflite(model)
    
    # Try to save in CoreML format if possible
    print("Converting model to CoreML format...")
    convert_to_coreml(model)
    
    print("Done!")

if __name__ == "__main__":
    main()
