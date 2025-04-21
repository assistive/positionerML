"""
Visualize the comparison between RNN and TCN models for IMU data processing.

This script creates visualizations for:
1. Side-by-side trajectory predictions during GPS outages
2. Error comparison across different outage durations
3. Computational performance metrics
4. Training convergence
"""

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter1d

# Set style
plt.style.use('seaborn-v0_8')
sns.set_context("paper", font_scale=1.4)

def load_history_files(rnn_history_path, tcn_history_path):
    """
    Load training history files for both models
    
    Args:
        rnn_history_path: Path to RNN training history CSV
        tcn_history_path: Path to TCN training history CSV
        
    Returns:
        rnn_history: DataFrame with RNN training history
        tcn_history: DataFrame with TCN training history
    """
    rnn_history = None
    tcn_history = None
    
    if os.path.exists(rnn_history_path):
        rnn_history = pd.read_csv(rnn_history_path)
    
    if os.path.exists(tcn_history_path):
        tcn_history = pd.read_csv(tcn_history_path)
    
    return rnn_history, tcn_history

def plot_training_convergence(rnn_history, tcn_history, output_path='training_convergence.png'):
    """
    Plot training convergence comparison
    
    Args:
        rnn_history: DataFrame with RNN training history
        tcn_history: DataFrame with TCN training history
        output_path: Path to save the output plot
    """
    if rnn_history is None or tcn_history is None:
        print("Warning: Training history not available for one or both models")
        return
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot position loss
    ax = axs[0, 0]
    if 'position_delta_loss' in rnn_history.columns:
        ax.plot(rnn_history['epoch'], rnn_history['position_delta_loss'], 
                label='RNN Training', color='blue', linestyle='-')
        ax.plot(rnn_history['epoch'], rnn_history['val_position_delta_loss'], 
                label='RNN Validation', color='blue', linestyle='--')
    
    if 'position_delta_loss' in tcn_history.columns:
        ax.plot(tcn_history['epoch'], tcn_history['position_delta_loss'], 
                label='TCN Training', color='red', linestyle='-')
        ax.plot(tcn_history['epoch'], tcn_history['val_position_delta_loss'], 
                label='TCN Validation', color='red', linestyle='--')
    
    ax.set_title('Position Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot velocity loss
    ax = axs[0, 1]
    if 'velocity_delta_loss' in rnn_history.columns:
        ax.plot(rnn_history['epoch'], rnn_history['velocity_delta_loss'], 
                label='RNN Training', color='blue', linestyle='-')
        ax.plot(rnn_history['epoch'], rnn_history['val_velocity_delta_loss'], 
                label='RNN Validation', color='blue', linestyle='--')
    
    if 'velocity_delta_loss' in tcn_history.columns:
        ax.plot(tcn_history['epoch'], tcn_history['velocity_delta_loss'], 
                label='TCN Training', color='red', linestyle='-')
        ax.plot(tcn_history['epoch'], tcn_history['val_velocity_delta_loss'], 
                label='TCN Validation', color='red', linestyle='--')
    
    ax.set_title('Velocity Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot position MAE
    ax = axs[1, 0]
    if 'position_delta_mae' in rnn_history.columns:
        ax.plot(rnn_history['epoch'], rnn_history['position_delta_mae'], 
                label='RNN Training', color='blue', linestyle='-')
        ax.plot(rnn_history['epoch'], rnn_history['val_position_delta_mae'], 
                label='RNN Validation', color='blue', linestyle='--')
    
    if 'position_delta_mae' in tcn_history.columns:
        ax.plot(tcn_history['epoch'], tcn_history['position_delta_mae'], 
                label='TCN Training', color='red', linestyle='-')
        ax.plot(tcn_history['epoch'], tcn_history['val_position_delta_mae'], 
                label='TCN Validation', color='red', linestyle='--')
    
    ax.set_title('Position MAE')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot velocity MAE
    ax = axs[1, 1]
    if 'velocity_delta_mae' in rnn_history.columns:
        ax.plot(rnn_history['epoch'], rnn_history['velocity_delta_mae'], 
                label='RNN Training', color='blue', linestyle='-')
        ax.plot(rnn_history['epoch'], rnn_history['val_velocity_delta_mae'], 
                label='RNN Validation', color='blue', linestyle='--')
    
    if 'velocity_delta_mae' in tcn_history.columns:
        ax.plot(tcn_history['epoch'], tcn_history['velocity_delta_mae'], 
                label='TCN Training', color='red', linestyle='-')
        ax.plot(tcn_history['epoch'], tcn_history['val_velocity_delta_mae'], 
                label='TCN Validation', color='red', linestyle='--')
    
    ax.set_title('Velocity MAE')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Training convergence plot saved to {output_path}")

def plot_side_by_side_trajectories(rnn_trajectory, tcn_trajectory, ground_truth, 
                                  outage_mask, output_path='side_by_side_trajectories.png'):
    """
    Plot side-by-side trajectory comparison
    
    Args:
        rnn_trajectory: RNN predicted trajectory
        tcn_trajectory: TCN predicted trajectory
        ground_truth: Ground truth trajectory
        outage_mask: Boolean mask where True indicates available GPS
        output_path: Path to save the output plot
    """
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig)
    
    # First row: 3D trajectories
    ax1 = fig.add_subplot(gs[0, 0:2], projection='3d')
    ax2 = fig.add_subplot(gs[0, 2], projection='3d')
    
    # Second row: 2D projections
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[1, 2])
    
    # Plot 3D full trajectory
    ax1.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2], 
             color='green', label='Ground Truth', linewidth=2)
    
    # Plot available GPS points
    ax1.scatter(ground_truth[outage_mask, 0], 
               ground_truth[outage_mask, 1], 
               ground_truth[outage_mask, 2],
               color='black', label='Available GPS', s=10, alpha=0.3)
    
    # Find outage segments
    outage_start = None
    for i in range(1, len(outage_mask)):
        if outage_mask[i-1] and not outage_mask[i]:  # Start of outage
            outage_start = i
        elif not outage_mask[i-1] and outage_mask[i]:  # End of outage
            if outage_start is not None:
                # Plot this outage segment
                segment_gt = ground_truth[outage_start:i]
                segment_rnn = rnn_trajectory[outage_start:i]
                segment_tcn = tcn_trajectory[outage_start:i]
                
                ax1.plot(segment_rnn[:, 0], segment_rnn[:, 1], segment_rnn[:, 2],
                        color='blue', linewidth=2, alpha=0.8)
                ax1.plot(segment_tcn[:, 0], segment_tcn[:, 1], segment_tcn[:, 2],
                        color='red', linewidth=2, alpha=0.8)
                
                outage_start = None
    
    ax1.set_title('Full Trajectory Comparison')
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.set_zlabel('Z (meters)')
    
    # Create custom legend
    gt_patch = mpatches.Patch(color='green', label='Ground Truth')
    rnn_patch = mpatches.Patch(color='blue', label='RNN Prediction')
    tcn_patch = mpatches.Patch(color='red', label='TCN Prediction')
    gps_patch = mpatches.Patch(color='black', label='Available GPS')
    ax1.legend(handles=[gt_patch, rnn_patch, tcn_patch, gps_patch], loc='upper right')
    
    # Plot 3D detailed trajectory (longest outage)
    # Find longest outage
    outage_lengths = []
    outage_segments = []
    outage_start = None
    
    for i in range(1, len(outage_mask)):
        if outage_mask[i-1] and not outage_mask[i]:  # Start of outage
            outage_start = i
        elif not outage_mask[i-1] and outage_mask[i]:  # End of outage
            if outage_start is not None:
                outage_length = i - outage_start
                outage_lengths.append(outage_length)
                outage_segments.append((outage_start, i))
                outage_start = None
    
    if outage_segments:
        longest_outage_idx = np.argmax(outage_lengths)
        longest_start, longest_end = outage_segments[longest_outage_idx]
        
        # Extract segments for longest outage
        segment_gt = ground_truth[longest_start:longest_end]
        segment_rnn = rnn_trajectory[longest_start:longest_end]
        segment_tcn = tcn_trajectory[longest_start:longest_end]
        
        # Plot detailed trajectory
        ax2.plot(segment_gt[:, 0], segment_gt[:, 1], segment_gt[:, 2], 
                color='green', label='Ground Truth', linewidth=2)
        ax2.plot(segment_rnn[:, 0], segment_rnn[:, 1], segment_rnn[:, 2],
                color='blue', label='RNN', linewidth=2)
        ax2.plot(segment_tcn[:, 0], segment_tcn[:, 1], segment_tcn[:, 2],
                color='red', label='TCN', linewidth=2)
        
        # Add arrows to show direction
        idx = len(segment_gt) // 2
        ax2.quiver(segment_gt[idx, 0], segment_gt[idx, 1], segment_gt[idx, 2],
                  segment_gt[idx+1, 0] - segment_gt[idx, 0],
                  segment_gt[idx+1, 1] - segment_gt[idx, 1],
                  segment_gt[idx+1, 2] - segment_gt[idx, 2],
                  color='green', length=2.0, normalize=True)
        
        # Add outage duration text
        duration_sec = outage_lengths[longest_outage_idx] * 0.01  # Assuming 100Hz
        ax2.text2D(0.05, 0.95, f"Outage Duration: {duration_sec:.1f}s", 
                  transform=ax2.transAxes, fontsize=12, color='black',
                  bbox=dict(facecolor='white', alpha=0.7))
        
        ax2.set_title('Longest GPS Outage')
    
    # Plot 2D projections (X-Y, X-Z, Y-Z)
    # X-Y projection
    ax3.plot(ground_truth[:, 0], ground_truth[:, 1], color='green', label='Ground Truth', linewidth=1.5)
    
    for start, end in outage_segments:
        segment_gt = ground_truth[start:end]
        segment_rnn = rnn_trajectory[start:end]
        segment_tcn = tcn_trajectory[start:end]
        
        ax3.plot(segment_gt[:, 0], segment_gt[:, 1], color='green', linewidth=1.5)
        ax3.plot(segment_rnn[:, 0], segment_rnn[:, 1], color='blue', linewidth=1.5)
        ax3.plot(segment_tcn[:, 0], segment_tcn[:, 1], color='red', linewidth=1.5)
    
    ax3.set_title('X-Y Projection')
    ax3.set_xlabel('X (meters)')
    ax3.set_ylabel('Y (meters)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # X-Z projection
    ax4.plot(ground_truth[:, 0], ground_truth[:, 2], color='green', label='Ground Truth', linewidth=1.5)
    
    for start, end in outage_segments:
        segment_gt = ground_truth[start:end]
        segment_rnn = rnn_trajectory[start:end]
        segment_tcn = tcn_trajectory[start:end]
        
        ax4.plot(segment_gt[:, 0], segment_gt[:, 2], color='green', linewidth=1.5)
        ax4.plot(segment_rnn[:, 0], segment_rnn[:, 2], color='blue', linewidth=1.5)
        ax4.plot(segment_tcn[:, 0], segment_tcn[:, 2], color='red', linewidth=1.5)
    
    ax4.set_title('X-Z Projection')
    ax4.set_xlabel('X (meters)')
    ax4.set_ylabel('Z (meters)')
    ax4.grid(True, alpha=0.3)
    
    # Y-Z projection
    ax5.plot(ground_truth[:, 1], ground_truth[:, 2], color='green', label='Ground Truth', linewidth=1.5)
    
    for start, end in outage_segments:
        segment_gt = ground_truth[start:end]
        segment_rnn = rnn_trajectory[start:end]
        segment_tcn = tcn_trajectory[start:end]
        
        ax5.plot(segment_gt[:, 1], segment_gt[:, 2], color='green', linewidth=1.5)
        ax5.plot(segment_rnn[:, 1], segment_rnn[:, 2], color='blue', linewidth=1.5)
        ax5.plot(segment_tcn[:, 1], segment_tcn[:, 2], color='red', linewidth=1.5)
    
    ax5.set_title('Y-Z Projection')
    ax5.set_xlabel('Y (meters)')
    ax5.set_ylabel('Z (meters)')
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Side-by-side trajectory plot saved to {output_path}")

def plot_error_metrics(rnn_results, tcn_results, output_path='error_metrics.png'):
    """
    Plot error metrics comparison
    
    Args:
        rnn_results: Dictionary with RNN evaluation results
        tcn_results: Dictionary with TCN evaluation results
        output_path: Path to save the output plot
    """
    fig, axs = plt.subplots(2, 2, figsize=(16, 14))
    
    # Plot mean error
    ax = axs[0, 0]
    models = ['RNN', 'TCN']
    mean_errors = [rnn_results['mean_error'], tcn_results['mean_error']]
    bars = ax.bar(models, mean_errors, color=['blue', 'red'])
    
    # Add percentage improvement
    improvement = (rnn_results['mean_error'] - tcn_results['mean_error']) / rnn_results['mean_error'] * 100
    ax.text(1, tcn_results['mean_error'] * 1.1, f"{improvement:.1f}% lower", 
           color='black', ha='center', fontsize=12)
    
    # Add values on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height * 1.02,
               f"{height:.2f}m", ha='center', va='bottom', fontsize=12)
    
    ax.set_title('Mean Position Error', fontsize=16)
    ax.set_ylabel('Error (meters)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Plot max error
    ax = axs[0, 1]
    max_errors = [rnn_results['max_error'], tcn_results['max_error']]
    bars = ax.bar(models, max_errors, color=['blue', 'red'])
    
    # Add percentage improvement
    improvement = (rnn_results['max_error'] - tcn_results['max_error']) / rnn_results['max_error'] * 100
    ax.text(1, tcn_results['max_error'] * 1.1, f"{improvement:.1f}% lower", 
           color='black', ha='center', fontsize=12)
    
    # Add values on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height * 1.02,
               f"{height:.2f}m", ha='center', va='bottom', fontsize=12)
    
    ax.set_title('Maximum Position Error', fontsize=16)
    ax.set_ylabel('Error (meters)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Plot error by outage duration
    ax = axs[1, 0]
    
    # Combine data from both models
    durations = sorted(set(list(rnn_results['mean_errors_by_duration'].keys()) + 
                          list(tcn_results['mean_errors_by_duration'].keys())))
    
    rnn_errors = [rnn_results['mean_errors_by_duration'].get(d, np.nan) for d in durations]
    tcn_errors = [tcn_results['mean_errors_by_duration'].get(d, np.nan) for d in durations]
    
    # Convert durations to seconds (assuming 100Hz)
    durations_sec = [d * 0.01 for
