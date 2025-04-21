"""
Compare the performance of different models for IMU-based dead reckoning,
including RNN/LSTM and TCN approaches.

This script evaluates models on the same test dataset and reports metrics
like position error during simulated GPS outages, as well as inference speed.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# Import model creation functions from other scripts
from train_rnn_model_deadreckoning import (
    create_dead_reckoning_model as create_rnn_model,
    simulate_gps_outages,
    dead_reckon as dead_reckon_rnn
)

from train_tcn_model import (
    create_tcn_dead_reckoning_model as create_tcn_model,
    dead_reckon as dead_reckon_tcn
)

def load_test_data(data_path, dataset_type='synthetic'):
    """
    Load test data for model evaluation
    
    Args:
        data_path: Path to data file or directory
        dataset_type: Type of dataset ('synthetic', 'oxiod', 'ronin', etc.)
        
    Returns:
        test_imu: IMU test data
        test_positions: Ground truth positions
    """
    if dataset_type == 'synthetic':
        # Load from saved numpy file
        if os.path.exists(data_path):
            data = np.load(data_path)
            return data['test_imu'], data['test_positions']
        else:
            # Generate synthetic data if no file exists
            from train_tcn_model import generate_synthetic_trajectory
            imu_data, position_data = generate_synthetic_trajectory(300, sampling_rate=100)
            
            # Split data
            train_ratio = 0.7
            val_ratio = 0.15
            test_ratio = 0.15
            
            # Train/test split
            train_idx = int(len(imu_data) * train_ratio)
            val_idx = train_idx + int(len(imu_data) * val_ratio)
            
            test_imu = imu_data[val_idx:]
            test_positions = position_data[val_idx:]
            
            # Save data for future use
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            np.savez(data_path, test_imu=test_imu, test_positions=test_positions)
            
            return test_imu, test_positions
    else:
        # Load from processed CSV file
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            
            # Extract features
            imu_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
            pos_cols = ['pos_x', 'pos_y', 'pos_z']
            
            # Check if all required columns exist
            if not all(col in df.columns for col in imu_cols + pos_cols):
                raise ValueError(f"Required columns not found in {data_path}")
            
            imu_data = df[imu_cols].values
            position_data = df[pos_cols].values
            
            # Split data
            train_ratio = 0.7
            val_ratio = 0.15
            test_ratio = 0.15
            
            # Train/test split
            train_idx = int(len(imu_data) * train_ratio)
            val_idx = train_idx + int(len(imu_data) * val_ratio)
            
            test_imu = imu_data[val_idx:]
            test_positions = position_data[val_idx:]
            
            return test_imu, test_positions
        else:
            raise FileNotFoundError(f"Data file not found: {data_path}")

def evaluate_model(model_path, model_type, test_imu, test_positions, sequence_length):
    """
    Evaluate a model on test data
    
    Args:
        model_path: Path to saved model
        model_type: Type of model ('rnn' or 'tcn')
        test_imu: IMU test data
        test_positions: Ground truth positions
        sequence_length: Sequence length for model
        
    Returns:
        results: Dictionary with evaluation results
    """
    # Load model
    model = load_model(model_path)
    
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
    all_errors = []
    outage_durations = []
    inference_times = []
    
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
                # Measure inference time
                start_time = tf.timestamp()
                
                if model_type == 'rnn':
                    dr_positions = dead_reckon_rnn(
                        outage_imu, initial_state, model, sequence_length)
                else:  # tcn
                    dr_positions = dead_reckon_tcn(
                        outage_imu, initial_state, model, sequence_length)
                
                end_time = tf.timestamp()
                inference_time = end_time - start_time
                inference_times.append(inference_time.numpy())
                
                # Fill in predicted positions
                predicted_positions[start:end] = dr_positions[:outage_length]
                
                # Calculate errors
                segment_errors = np.linalg.norm(
                    predicted_positions[start:end] - test_positions[start:end], axis=1)
                all_errors.extend(segment_errors)
                outage_durations.append(outage_length)
    
    # Calculate error statistics
    mean_error = np.mean(all_errors) if all_errors else 0
    max_error = np.max(all_errors) if all_errors else 0
    
    # Errors by outage duration
    duration_errors = {}
    for duration, errors in zip(outage_durations, all_errors):
        duration_key = duration // 100 * 100  # Round to nearest 100
        if duration_key not in duration_errors:
            duration_errors[duration_key] = []
        duration_errors[duration_key].append(errors)
    
    # Calculate mean error by duration
    mean_errors_by_duration = {
        k: np.mean(v) for k, v in duration_errors.items()
    }
    
    # Calculate inference speed (samples per second)
    inference_speed = np.mean(outage_durations) / np.mean(inference_times) if inference_times else 0
    
    # Plot trajectory with predictions
    plot_filename = f"trajectory_{model_type}.png"
    plot_trajectory_with_outages(test_positions, masked_positions, predicted_positions, 
                                plot_filename)
    
    # Return results
    results = {
        'model_type': model_type,
        'mean_error': mean_error,
        'max_error': max_error,
        'mean_errors_by_duration': mean_errors_by_duration,
        'inference_speed': inference_speed
    }
    
    return results

def plot_trajectory_with_outages(true_positions, masked_positions, predicted_positions, 
                               filename='trajectory.png'):
    """
    Plot the trajectory showing true path, GPS with outages, and predictions
    
    Args:
        true_positions: Ground truth position data [n_samples, 3]
        masked_positions: Position data with outages (NaN values)
        predicted_positions: Predicted positions during outages
        filename: Output file name
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
    
    # Plot predicted trajectory
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
    
    plt.savefig(filename)
    plt.close()

def plot_error_comparison(results, filename='error_comparison.png'):
    """
    Plot error comparison between models
    
    Args:
        results: List of result dictionaries from evaluate_model
        filename: Output file name
    """
    plt.figure(figsize=(15, 10))
    
    # Create subplots
    plt.subplot(2, 2, 1)
    model_types = [r['model_type'] for r in results]
    mean_errors = [r['mean_error'] for r in results]
    plt.bar(model_types, mean_errors)
    plt.title('Mean Position Error')
    plt.ylabel('Error (meters)')
    
    plt.subplot(2, 2, 2)
    max_errors = [r['max_error'] for r in results]
    plt.bar(model_types, max_errors)
    plt.title('Maximum Position Error')
    plt.ylabel('Error (meters)')
    
    plt.subplot(2, 2, 3)
    # Plot error by outage duration for each model
    for result in results:
        durations = sorted(result['mean_errors_by_duration'].keys())
        errors = [result['mean_errors_by_duration'][d] for d in durations]
        plt.plot(durations, errors, marker='o', label=result['model_type'])
    plt.title('Mean Error by Outage Duration')
    plt.xlabel('Outage Duration (samples)')
    plt.ylabel('Error (meters)')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    inference_speeds = [r['inference_speed'] for r in results]
    plt.bar(model_types, inference_speeds)
    plt.title('Inference Speed')
    plt.ylabel('Samples per second')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Compare IMU dead reckoning models')
    parser.add_argument('--data', type=str, default='./data/test_data.npz',
                       help='Path to test data file')
    parser.add_argument('--dataset_type', type=str, default='synthetic',
                       choices=['synthetic', 'oxiod', 'ronin', 'ridi', 'kitti'],
                       help='Type of dataset')
    parser.add_argument('--rnn_model', type=str, default='dead_reckoning_best.h5',
                       help='Path to RNN model')
    parser.add_argument('--tcn_model', type=str, default='dead_reckoning_tcn_best.h5',
                       help='Path to TCN model')
    parser.add_argument('--sequence_length', type=int, default=50,
                       help='Sequence length for models')
    
    args = parser.parse_args()
    
    # Load test data
    test_imu, test_positions = load_test_data(args.data, args.dataset_type)
    
    # Evaluate RNN model
    if os.path.exists(args.rnn_model):
        print(f"Evaluating RNN model: {args.rnn_model}")
        rnn_results = evaluate_model(args.rnn_model, 'rnn', test_imu, test_positions, 
                                    args.sequence_length)
        print(f"RNN Results:")
        print(f"  Mean error: {rnn_results['mean_error']:.2f} meters")
        print(f"  Max error: {rnn_results['max_error']:.2f} meters")
        print(f"  Inference speed: {rnn_results['inference_speed']:.2f} samples/second")
    else:
        print(f"RNN model not found: {args.rnn_model}")
        rnn_results = None
    
    # Evaluate TCN model
    if os.path.exists(args.tcn_model):
        print(f"Evaluating TCN model: {args.tcn_model}")
        tcn_results = evaluate_model(args.tcn_model, 'tcn', test_imu, test_positions, 
                                    args.sequence_length)
        print(f"TCN Results:")
        print(f"  Mean error: {tcn_results['mean_error']:.2f} meters")
        print(f"  Max error: {tcn_results['max_error']:.2f} meters")
        print(f"  Inference speed: {tcn_results['inference_speed']:.2f} samples/second")
    else:
        print(f"TCN model not found: {args.tcn_model}")
        tcn_results = None
    
    # Compare results
    results = []
    if rnn_results:
        results.append(rnn_results)
    if tcn_results:
        results.append(tcn_results)
    
    if len(results) > 1:
        print("Plotting comparison...")
        plot_error_comparison(results)
        
        # Print improvement statistics
        rnn_error = rnn_results['mean_error']
        tcn_error = tcn_results['mean_error']
        
        error_improvement = (rnn_error - tcn_error) / rnn_error * 100
        
        rnn_speed = rnn_results['inference_speed']
        tcn_speed = tcn_results['inference_speed']
        
        speed_improvement = (tcn_speed - rnn_speed) / rnn_speed * 100
        
        print(f"TCN improves mean error by {error_improvement:.2f}%")
        print(f"TCN improves inference speed by {speed_improvement:.2f}%")

if __name__ == "__main__":
    main()
