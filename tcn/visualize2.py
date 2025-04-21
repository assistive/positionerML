"""
Comprehensive IMU data visualization script for RNN model outputs.
This script creates visualizations for comparing raw and smoothed IMU data,
dead reckoning trajectories, and model performance metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
from sklearn.metrics import mean_squared_error
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from tensorflow.keras.models import load_model
import tensorflow as tf
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def load_imu_data(file_path):
    """
    Load IMU data from CSV file
    
    Args:
        file_path: Path to CSV with IMU data
        
    Returns:
        DataFrame with IMU data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
        
    df = pd.read_csv(file_path)
    
    # Verify expected columns exist
    expected_cols = ['timestamp']
    imu_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
    expected_cols.extend(imu_cols)
    
    # Check if position data is available
    position_cols = ['position_x', 'position_y', 'position_z']
    has_position = all(col in df.columns for col in position_cols)
    
    # Check if smoothed data is available
    smoothed_cols = ['smoothed_accel_x', 'smoothed_accel_y', 'smoothed_accel_z',
                     'smoothed_gyro_x', 'smoothed_gyro_y', 'smoothed_gyro_z']
    has_smoothed = all(col in df.columns for col in smoothed_cols)
    
    print(f"Loaded data with {len(df)} rows")
    print(f"Position data available: {has_position}")
    print(f"Smoothed data available: {has_smoothed}")
    
    return df, has_position, has_smoothed

def normalize_data(df, cols):
    """
    Normalize selected columns in DataFrame
    
    Args:
        df: DataFrame with IMU data
        cols: List of columns to normalize
        
    Returns:
        DataFrame with normalized columns
    """
    result = df.copy()
    for col in cols:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            result[col] = (df[col] - mean) / std
    
    return result

def plot_imu_comparison(df, raw_cols, smoothed_cols, title, output_path=None):
    """
    Plot comparison between raw and smoothed IMU data
    
    Args:
        df: DataFrame with IMU data
        raw_cols: List of raw IMU column names
        smoothed_cols: List of smoothed IMU column names
        title: Plot title
        output_path: Optional path to save the figure
    """
    n_cols = len(raw_cols)
    fig, axes = plt.subplots(n_cols, 1, figsize=(15, 3*n_cols), sharex=True)
    
    if n_cols == 1:
        axes = [axes]  # Make it iterable if only one axis
    
    for i, (raw_col, smooth_col) in enumerate(zip(raw_cols, smoothed_cols)):
        if raw_col in df.columns and smooth_col in df.columns:
            axes[i].plot(df['timestamp'], df[raw_col], 'b-', alpha=0.6, label='Raw')
            axes[i].plot(df['timestamp'], df[smooth_col], 'g-', linewidth=2, label='Smoothed')
            axes[i].set_ylabel(raw_col.replace('accel_', 'Accel ').replace('gyro_', 'Gyro '))
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {output_path}")
    
    plt.show()

def plot_3d_trajectory(df, output_path=None):
    """
    Plot 3D trajectory from position data
    
    Args:
        df: DataFrame with position data
        output_path: Optional path to save the figure
    """
    if not all(col in df.columns for col in ['position_x', 'position_y', 'position_z']):
        print("Cannot plot trajectory: position data not available")
        return
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    ax.plot(df['position_x'], df['position_y'], df['position_z'], 'b-', linewidth=2)
    
    # Add markers for start and end points
    ax.scatter(df['position_x'].iloc[0], df['position_y'].iloc[0], df['position_z'].iloc[0], 
               c='green', marker='o', s=100, label='Start')
    ax.scatter(df['position_x'].iloc[-1], df['position_y'].iloc[-1], df['position_z'].iloc[-1], 
               c='red', marker='o', s=100, label='End')
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory')
    ax.legend()
    
    # Make equal aspect ratio
    max_range = max([
        df['position_x'].max() - df['position_x'].min(),
        df['position_y'].max() - df['position_y'].min(),
        df['position_z'].max() - df['position_z'].min()
    ])
    
    mid_x = (df['position_x'].max() + df['position_x'].min()) / 2
    mid_y = (df['position_y'].max() + df['position_y'].min()) / 2
    mid_z = (df['position_z'].max() + df['position_z'].min()) / 2
    
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved trajectory plot to {output_path}")
    
    plt.show()

def plot_dead_reckoning_comparison(df, true_trajectory, dead_reckoning_trajectory, gps_outage_mask=None, output_path=None):
    """
    Plot comparison between true trajectory and dead reckoning during GPS outages
    
    Args:
        df: DataFrame with timestamp information
        true_trajectory: Array of true positions [n_samples, 3]
        dead_reckoning_trajectory: Array of dead reckoning positions [n_samples, 3]
        gps_outage_mask: Boolean mask where False indicates GPS outage
        output_path: Optional path to save the figure
    """
    # Create figure with 2 subplots: 2D top view and 3D view
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    
    # 2D top view (X-Y plane)
    ax1 = plt.subplot(gs[0])
    ax1.plot(true_trajectory[:, 0], true_trajectory[:, 1], 'g-', linewidth=2, label='Ground Truth')
    
    if gps_outage_mask is not None:
        # Plot available GPS points
        ax1.scatter(true_trajectory[gps_outage_mask, 0], true_trajectory[gps_outage_mask, 1], 
                    c='blue', s=20, label='Available GPS')
                    
        # Plot dead reckoning segments during outages
        outage_start = None
        for i in range(1, len(gps_outage_mask)):
            if gps_outage_mask[i-1] and not gps_outage_mask[i]:  # Start of outage
                outage_start = i
            elif not gps_outage_mask[i-1] and gps_outage_mask[i]:  # End of outage
                if outage_start is not None:
                    # Plot this outage segment
                    segment = dead_reckoning_trajectory[outage_start:i]
                    ax1.plot(segment[:, 0], segment[:, 1], 'r-', linewidth=2)
                    outage_start = None
    else:
        # No outage mask provided, just plot the full DR trajectory
        ax1.plot(dead_reckoning_trajectory[:, 0], dead_reckoning_trajectory[:, 1], 
                 'r--', linewidth=2, label='Dead Reckoning')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Top View (X-Y Plane)')
    ax1.grid(True)
    ax1.legend()
    ax1.set_aspect('equal')
    
    # 3D view
    ax2 = plt.subplot(gs[1], projection='3d')
    ax2.plot(true_trajectory[:, 0], true_trajectory[:, 1], true_trajectory[:, 2], 
             'g-', linewidth=2, label='Ground Truth')
             
    if gps_outage_mask is not None:
        # Plot available GPS points
        ax2.scatter(true_trajectory[gps_outage_mask, 0], 
                  true_trajectory[gps_outage_mask, 1], 
                  true_trajectory[gps_outage_mask, 2],
                  c='blue', s=20, label='Available GPS')
                  
        # Plot dead reckoning segments during outages
        outage_start = None
        for i in range(1, len(gps_outage_mask)):
            if gps_outage_mask[i-1] and not gps_outage_mask[i]:  # Start of outage
                outage_start = i
            elif not gps_outage_mask[i-1] and gps_outage_mask[i]:  # End of outage
                if outage_start is not None:
                    # Plot this outage segment
                    segment = dead_reckoning_trajectory[outage_start:i]
                    ax2.plot(segment[:, 0], segment[:, 1], segment[:, 2], 'r-', linewidth=2)
                    outage_start = None
    else:
        # No outage mask provided, just plot the full DR trajectory
        ax2.plot(dead_reckoning_trajectory[:, 0], dead_reckoning_trajectory[:, 1], dead_reckoning_trajectory[:, 2], 
                 'r--', linewidth=2, label='Dead Reckoning')
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title('3D Trajectory View')
    ax2.legend()
    
    plt.suptitle('Dead Reckoning Performance During GPS Outages', fontsize=16)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved dead reckoning comparison to {output_path}")
    
    plt.show()

def calculate_error_metrics(true_traj, dr_traj, gps_outage_mask=None):
    """
    Calculate error metrics between true and dead reckoning trajectories
    
    Args:
        true_traj: Array of true positions [n_samples, 3]
        dr_traj: Array of dead reckoning positions [n_samples, 3]
        gps_outage_mask: Boolean mask where False indicates GPS outage
        
    Returns:
        Dictionary of error metrics
    """
    if gps_outage_mask is not None:
        # Calculate errors only during outages
        outage_indices = np.where(~gps_outage_mask)[0]
        
        if len(outage_indices) == 0:
            print("No GPS outages found in mask")
            return {}
            
        true_outage = true_traj[outage_indices]
        dr_outage = dr_traj[outage_indices]
        
        # Calculate position errors
        position_errors = np.linalg.norm(dr_outage - true_outage, axis=1)
        
        # Find outage segments
        segments = []
        current_segment = []
        
        for i in range(1, len(outage_indices)):
            if outage_indices[i] == outage_indices[i-1] + 1:
                if not current_segment:
                    current_segment = [outage_indices[i-1]]
                current_segment.append(outage_indices[i])
            else:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
        
        if current_segment:
            segments.append(current_segment)
            
        # Calculate error growth per segment
        error_growth_rates = []
        for segment in segments:
            if len(segment) > 10:  # Only consider segments with reasonable length
                segment_errors = position_errors[:len(segment)]
                # Linear regression to find error growth rate
                times = np.arange(len(segment_errors)) / 100  # Assuming 100Hz
                if len(times) > 1:
                    error_growth = np.polyfit(times, segment_errors, 1)[0]  # Slope
                    error_growth_rates.append(error_growth)
        
        metrics = {
            'mean_error': np.mean(position_errors),
            'max_error': np.max(position_errors),
            'rmse': np.sqrt(mean_squared_error(true_outage, dr_outage)),
            'median_error': np.median(position_errors),
            'error_growth_rate': np.mean(error_growth_rates) if error_growth_rates else None
        }
    else:
        # Calculate errors over the entire trajectory
        position_errors = np.linalg.norm(dr_traj - true_traj, axis=1)
        
        metrics = {
            'mean_error': np.mean(position_errors),
            'max_error': np.max(position_errors),
            'rmse': np.sqrt(mean_squared_error(true_traj, dr_traj)),
            'median_error': np.median(position_errors)
        }
    
    return metrics

def plot_error_distribution(true_traj, dr_traj, gps_outage_mask=None, output_path=None):
    """
    Plot error distribution and growth during GPS outages
    
    Args:
        true_traj: Array of true positions [n_samples, 3]
        dr_traj: Array of dead reckoning positions [n_samples, 3]
        gps_outage_mask: Boolean mask where False indicates GPS outage
        output_path: Optional path to save the figure
    """
    if gps_outage_mask is not None:
        # Find GPS outage segments
        outage_segments = []
        outage_start = None
        
        for i in range(1, len(gps_outage_mask)):
            if gps_outage_mask[i-1] and not gps_outage_mask[i]:  # Start of outage
                outage_start = i
            elif not gps_outage_mask[i-1] and gps_outage_mask[i]:  # End of outage
                if outage_start is not None:
                    outage_segments.append((outage_start, i))
                    outage_start = None
        
        if outage_start is not None:  # Handle outage at the end
            outage_segments.append((outage_start, len(gps_outage_mask)))
            
        if not outage_segments:
            print("No GPS outages found")
            return
            
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot error growth over time for each outage
        for start, end in outage_segments:
            if end - start < 5:  # Skip very short outages
                continue
                
            segment_true = true_traj[start:end]
            segment_dr = dr_traj[start:end]
            
            # Calculate position errors
            errors = np.linalg.norm(segment_dr - segment_true, axis=1)
            
            # Plot error growth
            times = np.arange(len(errors)) / 100  # Assuming 100Hz
            ax1.plot(times, errors, alpha=0.7, linewidth=1.5)
        
        ax1.set_xlabel('Time Since Outage Start (s)')
        ax1.set_ylabel('Position Error (m)')
        ax1.set_title('Error Growth During GPS Outages')
        ax1.grid(True)
        
        # Combine all outage errors for distribution plot
        all_errors = []
        for start, end in outage_segments:
            segment_true = true_traj[start:end]
            segment_dr = dr_traj[start:end]
            errors = np.linalg.norm(segment_dr - segment_true, axis=1)
            all_errors.extend(errors)
            
        # Plot error distribution
        sns.histplot(all_errors, kde=True, ax=ax2)
        ax2.axvline(np.mean(all_errors), color='r', linestyle='--', label=f'Mean: {np.mean(all_errors):.2f}m')
        ax2.axvline(np.median(all_errors), color='g', linestyle='--', label=f'Median: {np.median(all_errors):.2f}m')
        ax2.set_xlabel('Position Error (m)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Error Distribution During GPS Outages')
        ax2.legend()
        
        plt.suptitle('Dead Reckoning Error Analysis', fontsize=16)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved error analysis to {output_path}")
        
        plt.show()
    else:
        # No outage mask, calculate errors over entire trajectory
        errors = np.linalg.norm(dr_traj - true_traj, axis=1)
        
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True)
        plt.axvline(np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.2f}m')
        plt.axvline(np.median(errors), color='g', linestyle='--', label=f'Median: {np.median(errors):.2f}m')
        plt.xlabel('Position Error (m)')
        plt.ylabel('Frequency')
        plt.title('Position Error Distribution')
        plt.legend()
        plt.grid(True)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved error analysis to {output_path}")
        
        plt.show()

def visualize_model_performance(model_path, test_data, output_dir=None):
    """
    Visualize model performance by comparing predictions with ground truth
    
    Args:
        model_path: Path to trained TensorFlow model
        test_data: Dictionary with test inputs and targets
        output_dir: Optional directory to save output figures
    """
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
        
    # Load model
    try:
        model = load_model(model_path)
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Make predictions
    X_test = test_data['X_test']
    y_true = test_data['y_test']
    
    y_pred = model.predict(X_test)
    
    # Handle different model output formats
    if isinstance(y_pred, list):
        # Multiple outputs (e.g. position_delta and velocity_delta)
        position_pred = y_pred[0]
        position_true = y_true[0] if isinstance(y_true, list) else y_true
    else:
        position_pred = y_pred
        position_true = y_true
    
    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot predictions vs ground truth for each dimension
    n_dims = position_pred.shape[1]
    dim_names = ['X', 'Y', 'Z'][:n_dims]
    
    plt.figure(figsize=(15, 5 * n_dims))
    
    for i in range(n_dims):
        plt.subplot(n_dims, 1, i+1)
        plt.plot(position_true[:100, i], 'g-', label='Ground Truth')
        plt.plot(position_pred[:100, i], 'r--', label='Prediction')
        plt.xlabel('Sample')
        plt.ylabel(f'{dim_names[i]} Position Delta')
        plt.title(f'{dim_names[i]} Position Change Prediction')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'prediction_comparison.png'), dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Plot error distribution
    errors = np.linalg.norm(position_pred - position_true, axis=1)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.axvline(np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.4f}')
    plt.xlabel('Prediction Error Magnitude')
    plt.ylabel('Frequency')
    plt.title('Model Prediction Error Distribution')
    plt.legend()
    plt.grid(True)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'prediction_error_distribution.png'), dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print error metrics
    print("Prediction Error Metrics:")
    print(f"  Mean Error: {np.mean(errors):.4f}")
    print(f"  Median Error: {np.median(errors):.4f}")
    print(f"  Max Error: {np.max(errors):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(position_true, position_pred)):.4f}")

def create_dashboard(df, model=None, output_dir=None):
    """
    Create a comprehensive dashboard with multiple visualizations
    
    Args:
        df: DataFrame with IMU and position data
        model: Optional loaded model for predictions
        output_dir: Directory to save output figures
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create base figure with grid layout
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 3, figure=fig)
    
    # 1. Raw vs Smoothed Accelerometer Data
    ax1 = fig.add_subplot(gs[0, 0])
    if all(col in df.columns for col in ['accel_x', 'smoothed_accel_x']):
        ax1.plot(df['timestamp'], df['accel_x'], 'b-', alpha=0.6, label='Raw')
        ax1.plot(df['timestamp'], df['smoothed_accel_x'], 'g-', linewidth=2, label='Smoothed')
        ax1.set_ylabel('Acceleration X (m/s²)')
        ax1.set_title('Accelerometer X: Raw vs Smoothed')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'Accelerometer data not available', 
                 horizontalalignment='center', verticalalignment='center')
    
    # 2. Raw vs Smoothed Gyroscope Data
    ax2 = fig.add_subplot(gs[0, 1])
    if all(col in df.columns for col in ['gyro_z', 'smoothed_gyro_z']):
        ax2.plot(df['timestamp'], df['gyro_z'], 'b-', alpha=0.6, label='Raw')
        ax2.plot(df['timestamp'], df['smoothed_gyro_z'], 'g-', linewidth=2, label='Smoothed')
        ax2.set_ylabel('Angular Velocity Z (rad/s)')
        ax2.set_title('Gyroscope Z: Raw vs Smoothed')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Gyroscope data not available', 
                 horizontalalignment='center', verticalalignment='center')
    
    # 3. 3D Trajectory
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    if all(col in df.columns for col in ['position_x', 'position_y', 'position_z']):
        ax3.plot(df['position_x'], df['position_y'], df['position_z'], 'b-', linewidth=2)
        
        # Add markers for start and end points
        ax3.scatter(df['position_x'].iloc[0], df['position_y'].iloc[0], df['position_z'].iloc[0], 
                   c='green', marker='o', s=100, label='Start')
        ax3.scatter(df['position_x'].iloc[-1], df['position_y'].iloc[-1], df['position_z'].iloc[-1], 
                   c='red', marker='o', s=100, label='End')
        
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_zlabel('Z (m)')
        ax3.set_title('3D Trajectory')
        ax3.legend()
    else:
        ax3.text2D(0.5, 0.5, 'Position data not available', 
                  horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
    
    # 4. Acceleration Magnitude
    ax4 = fig.add_subplot(gs[1, 0])
    if all(col in df.columns for col in ['accel_x', 'accel_y', 'accel_z']):
        # Calculate acceleration magnitude
        acc_mag = np.sqrt(df['accel_x']**2 + df['accel_y']**2 + df['accel_z']**2)
        ax4.plot(df['timestamp'], acc_mag, 'b-', linewidth=1.5)
        ax4.axhline(y=9.81, color='r', linestyle='--', label='1g')
        ax4.set_ylabel('Acceleration Magnitude (m/s²)')
        ax4.set_title('Acceleration Magnitude')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Accelerometer data not available', 
                 horizontalalignment='center', verticalalignment='center')
    
    # 5. Angular Velocity Magnitude
    ax5 = fig.add_subplot(gs[1, 1])
    if all(col in df.columns for col in ['gyro_x', 'gyro_y', 'gyro_z']):
        # Calculate angular velocity magnitude
        gyro_mag = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)
        ax5.plot(df['timestamp'], gyro_mag, 'b-', linewidth=1.5)
        ax5.set_ylabel('Angular Velocity Magnitude (rad/s)')
        ax5.set_title('Angular Velocity Magnitude')
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'Gyroscope data not available', 
                 horizontalalignment='center', verticalalignment='center')
    
    # 6. Trajectory Top View (X-Y Plane)
    ax6 = fig.add_subplot(gs[1, 2])
    if all(col in df.columns for col in ['position_x', 'position_y']):
        ax6.plot(df['position_x'], df['position_y'], 'b-', linewidth=2)
        ax6.scatter(df['position_x'].iloc[0], df['position_y'].iloc[0], 
                   c='green', marker='o', s=100, label='Start')
        ax6.scatter(df['position_x'].iloc[-1], df['position_y'].iloc[-1], 
                   c='red', marker='o', s=100, label='End')
        ax6.set_xlabel('X (m)')
        ax6.set_ylabel('Y (m)')
        ax6.set_title('Trajectory Top View (X-Y Plane)')
        ax6.grid(True)
        ax6.set_aspect('equal')
        ax6.legend()
    else:
        ax6.text(0.5, 0.5, 'Position data not available', 
                 horizontalalignment='center', verticalalignment='center')
    
    # 7. IMU Jerk Analysis
    ax7 = fig.add_subplot(gs[2, 0])
    if all(col in df.columns for col in ['accel_x', 'accel_y', 'accel_z']):
        # Calculate jerk (derivative of acceleration)
        jerk_x = np.gradient(df['accel_x'].values, df['timestamp'].values)
        jerk_y = np.gradient(df['accel_y'].values, df['timestamp'].values)
        jerk_z = np.gradient(df['accel_z'].values, df['timestamp'].values)
        
        # Calculate jerk magnitude
        jerk_mag = np.sqrt(jerk_x**2 + jerk_y**2 + jerk_z**2)
        
        ax7.plot(df['timestamp'], jerk_mag, 'b-', linewidth=1.5)
        ax7.set_ylabel('Jerk Magnitude (m/s³)')
        ax7.set_title('Jerk Magnitude (Derivative of Acceleration)')
        ax7.grid(True, alpha=0.3)
    else:
        ax7.text(0.5, 0.5, 'Accelerometer data not available', 
                 horizontalalignment='center', verticalalignment='center')
    
    # 8. Model Performance or Additional Analysis
    ax8 = fig.add_subplot(gs[2, 1:])
    if model is not None:
        ax8.text(0.5, 0.5, 'Model performance visualization would go here', 
                 horizontalalignment='center', verticalalignment='center')
    else:
        # Create motion events analysis if model not provided
        if all(col in df.columns for col in ['accel_x', 'accel_y', 'accel_z']):
            # Use acceleration magnitude to detect motion events
            acc_mag = np.sqrt(df['accel_x']**2 + df['accel_y']**2 + df['accel_z']**2)
            
            # Simple threshold-based event detection
            acc_threshold = 2.0  # 2 m/s² above gravity
            motion_events = acc_mag > (9.81 + acc_threshold)
            
            # Plot acceleration magnitude with highlighted events
            ax8.plot(df['timestamp'], acc_mag, 'b-', linewidth=1.5)
            ax8.axhline(y=9.81 + acc_threshold, color='r', linestyle='--', label='Event Threshold')
            
            # Highlight motion events
            event_regions = df[motion_events]['timestamp'].values
            for t in event_regions:
                ax8.axvline(t, color='r', alpha=0.1)
            
            ax8.set_ylabel('Acceleration Magnitude (m/s²)')
            ax8.set_title('Motion Event Detection')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        else:
            ax8.text(0.5, 0.5, 'Additional analysis not available without model or IMU data', 
                     horizontalalignment='center', verticalalignment='center')
    
    plt.suptitle('IMU Data Analysis Dashboard', fontsize=16)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'dashboard.png'), dpi=300, bbox_inches='tight')
        print(f"Saved dashboard to {os.path.join(output_dir, 'dashboard.png')}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize IMU data and model performance')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV file with IMU data')
    parser.add_argument('--model', type=str, help='Path to trained model file (.h5 or .tflite)')
    parser.add_argument('--output', type=str, help='Directory to save output visualizations')
    parser.add_argument('--mode', type=str, default='all', 
                        choices=['smoothing', 'trajectory', 'dead_reckoning', 'dashboard', 'all'],
                        help='Visualization mode')
    args = parser.parse_args()
    
    # Load data
    df, has_position, has_smoothed = load_imu_data(args.data)
    
    # Create visualizations based on mode
    if args.mode in ['smoothing', 'all'] and has_smoothed:
        # Visualize IMU smoothing
        accel_cols = ['accel_x', 'accel_y', 'accel_z']
        smoothed_accel_cols = ['smoothed_accel_x', 'smoothed_accel_y', 'smoothed_accel_z']
        plot_imu_comparison(df, accel_cols, smoothed_accel_cols, 
                           'Accelerometer: Raw vs Smoothed', 
                           output_path=os.path.join(args.output, 'accel_comparison.png') if args.output else None)
        
        gyro_cols = ['gyro_x', 'gyro_y', 'gyro_z']
        smoothed_gyro_cols = ['smoothed_gyro_x', 'smoothed_gyro_y', 'smoothed_gyro_z']
        plot_imu_comparison(df, gyro_cols, smoothed_gyro_cols, 
                           'Gyroscope: Raw vs Smoothed', 
                           output_path=os.path.join(args.output, 'gyro_comparison.png') if args.output else None)
    
    if args.mode in ['trajectory', 'all'] and has_position:
        # Visualize 3D trajectory
        plot_3d_trajectory(df, 
                          output_path=os.path.join(args.output, 'trajectory_3d.png') if args.output else None)
    
    if args.mode in ['dashboard', 'all']:
        # Create comprehensive dashboard
        create_dashboard(df, 
                        output_dir=args.output if args.output else None)
    
    if args.mode in ['dead_reckoning', 'all'] and has_position:
        # For dead reckoning visualization, we need to simulate GPS outages
        # since we don't have real outage data in the CSV
        
        # Extract positions
        true_positions = df[['position_x', 'position_y', 'position_z']].values
        
        # Simulate 3 random GPS outages
        num_samples = len(df)
        gps_outage_mask = np.ones(num_samples, dtype=bool)  # True = GPS available
        
        for _ in range(3):
            # Outage of 3-5 seconds (300-500 samples at 100Hz)
            outage_length = np.random.randint(300, 500)
            outage_start = np.random.randint(0, num_samples - outage_length)
            gps_outage_mask[outage_start:outage_start+outage_length] = False
        
        # Simple dead reckoning simulation (just add increasing error)
        dead_reckoning_positions = true_positions.copy()
        
        for i in range(num_samples):
            if not gps_outage_mask[i]:
                # During outage, add some error that grows with time
                if i > 0 and not gps_outage_mask[i-1]:
                    # Existing outage, increase error
                    error_factor = 0.01  # 1cm error growth per sample
                    error = np.random.normal(0, error_factor * np.sqrt(i - np.argmin(gps_outage_mask[:i])), 3)
                    dead_reckoning_positions[i] = dead_reckoning_positions[i-1] + (true_positions[i] - true_positions[i-1]) + error
                else:
                    # Start of outage
                    dead_reckoning_positions[i] = true_positions[i]
        
        # Visualize dead reckoning performance
        plot_dead_reckoning_comparison(df, true_positions, dead_reckoning_positions, gps_outage_mask,
                                      output_path=os.path.join(args.output, 'dead_reckoning.png') if args.output else None)
        
        # Analyze errors
        plot_error_distribution(true_positions, dead_reckoning_positions, gps_outage_mask,
                               output_path=os.path.join(args.output, 'error_analysis.png') if args.output else None)
        
        metrics = calculate_error_metrics(true_positions, dead_reckoning_positions, gps_outage_mask)
        print("Dead Reckoning Error Metrics:")
        for key, value in metrics.items():
            if value is not None:
                print(f"  {key}: {value:.4f}")

if __name__ == "__main__":
    main()
