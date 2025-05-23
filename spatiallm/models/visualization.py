# spatiallm/utils/visualization.py

"""
Visualization utilities for SpatialLM training and evaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import torch
from pathlib import Path

def plot_training_curves(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Plot training and validation curves.
    
    Args:
        history: Dictionary containing training history
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    if 'train_loss' in history and 'val_loss' in history:
        ax = axes[0, 0]
        ax.plot(history['train_loss'], label='Training Loss')
        ax.plot(history['val_loss'], label='Validation Loss')
        ax.set_title('Loss Curves')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Perplexity curves
    if 'train_perplexity' in history and 'val_perplexity' in history:
        ax = axes[0, 1]
        ax.plot(history['train_perplexity'], label='Training Perplexity')
        ax.plot(history['val_perplexity'], label='Validation Perplexity')
        ax.set_title('Perplexity Curves')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Perplexity')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Spatial loss curves
    if 'train_spatial_loss' in history and 'val_spatial_loss' in history:
        ax = axes[1, 0]
        ax.plot(history['train_spatial_loss'], label='Training Spatial Loss')
        ax.plot(history['val_spatial_loss'], label='Validation Spatial Loss')
        ax.set_title('Spatial Loss Curves')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Spatial Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Learning rate
    if 'learning_rate' in history:
        ax = axes[1, 1]
        ax.plot(history['learning_rate'])
        ax.set_title('Learning Rate Schedule')
        ax.set_xlabel('Step')
        ax.set_ylabel('Learning Rate')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_spatial_predictions(true_coords: np.ndarray, 
                           pred_coords: np.ndarray,
                           sample_indices: Optional[List[int]] = None,
                           save_path: Optional[str] = None):
    """
    Plot spatial coordinate predictions vs ground truth.
    
    Args:
        true_coords: Ground truth coordinates [n_samples, 3]
        pred_coords: Predicted coordinates [n_samples, 3]
        sample_indices: Optional indices to highlight
        save_path: Optional path to save the plot
    """
    fig = plt.figure(figsize=(18, 6))
    
    # 3D scatter plot
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(true_coords[:, 0], true_coords[:, 1], true_coords[:, 2], 
               c='blue', label='Ground Truth', alpha=0.6, s=20)
    ax1.scatter(pred_coords[:, 0], pred_coords[:, 1], pred_coords[:, 2], 
               c='red', label='Predictions', alpha=0.6, s=20)
    
    if sample_indices:
        ax1.scatter(true_coords[sample_indices, 0], 
                   true_coords[sample_indices, 1], 
                   true_coords[sample_indices, 2],
                   c='green', s=100, marker='^', label='Highlighted')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Spatial Predictions')
    ax1.legend()
    
    # 2D projections
    # X-Y projection
    ax2 = fig.add_subplot(132)
    ax2.scatter(true_coords[:, 0], true_coords[:, 1], c='blue', label='Ground Truth', alpha=0.6)
    ax2.scatter(pred_coords[:, 0], pred_coords[:, 1], c='red', label='Predictions', alpha=0.6)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('X-Y Projection')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Error distribution
    ax3 = fig.add_subplot(133)
    errors = np.linalg.norm(true_coords - pred_coords, axis=1)
    ax3.hist(errors, bins=30, alpha=0.7, edgecolor='black')
    ax3.axvline(np.mean(errors), color='red', linestyle='--', 
               label=f'Mean: {np.mean(errors):.3f}')
    ax3.axvline(np.median(errors), color='green', linestyle='--', 
               label=f'Median: {np.median(errors):.3f}')
    ax3.set_xlabel('Euclidean Distance Error')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Error Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_attention_weights(attention_weights: torch.Tensor, 
                         tokens: List[str],
                         layer_idx: int = -1,
                         head_idx: int = 0,
                         save_path: Optional[str] = None):
    """
    Visualize attention weights.
    
    Args:
        attention_weights: Attention weights tensor
        tokens: List of tokens
        layer_idx: Which layer to visualize (-1 for last)
        head_idx: Which attention head to visualize
        save_path: Optional path to save the plot
    """
    # Extract attention weights for specific layer and head
    if attention_weights.dim() == 5:  # [batch, layer, head, seq, seq]
        attn = attention_weights[0, layer_idx, head_idx].cpu().numpy()
    else:
        attn = attention_weights[0, head_idx].cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens, 
               cmap='Blues', cbar=True, square=True)
    
    plt.title(f'Attention Weights - Layer {layer_idx}, Head {head_idx}')
    plt.xlabel('Keys')
    plt.ylabel('Queries')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_spatial_attention_influence(model, tokenizer, text: str, 
                                   coordinate_variations: List[Tuple[float, float, float]],
                                   save_path: Optional[str] = None):
    """
    Visualize how spatial coordinates influence model predictions.
    
    Args:
        model: The SpatialLM model
        tokenizer: The tokenizer
        text: Input text
        coordinate_variations: List of coordinate variations to test
        save_path: Optional path to save the plot
    """
    model.eval()
    
    # Tokenize text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    
    # Get predictions for each coordinate variation
    predictions = []
    
    with torch.no_grad():
        for coords in coordinate_variations:
            spatial_coords = torch.tensor([coords], dtype=torch.float)
            
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                spatial_coordinates=spatial_coords,
                output_hidden_states=True
            )
            
            # Get the last hidden state
            hidden_state = outputs.hidden_states[-1][:, -1, :].cpu().numpy()
            predictions.append(hidden_state.flatten())
    
    predictions = np.array(predictions)
    
    # Compute pairwise distances
    n_coords = len(coordinate_variations)
    distances = np.zeros((n_coords, n_coords))
    
    for i in range(n_coords):
        for j in range(n_coords):
            distances[i, j] = np.linalg.norm(predictions[i] - predictions[j])
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap of representation distances
    labels = [f"({x:.1f}, {y:.1f}, {z:.1f})" for x, y, z in coordinate_variations]
    sns.heatmap(distances, xticklabels=labels, yticklabels=labels, 
               cmap='viridis', cbar=True, square=True)
    
    plt.title('Representation Distance for Different Spatial Coordinates')
    plt.xlabel('Coordinates')
    plt.ylabel('Coordinates')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_evaluation_report(results: Dict[str, Any], 
                           output_path: str,
                           include_plots: bool = True):
    """
    Create a comprehensive evaluation report.
    
    Args:
        results: Evaluation results dictionary
        output_path: Path to save the report
        include_plots: Whether to include plots in the report
    """
    report_dir = Path(output_path)
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Create markdown report
    report_content = f"""# SpatialLM Evaluation Report

## Model Information
- Model: {results.get('model_name', 'Unknown')}
- Evaluation Date: {results.get('date', 'Unknown')}
- Test Dataset: {results.get('dataset', 'Unknown')}

## Language Modeling Performance
- Perplexity: {results.get('perplexity', 'N/A'):.2f}
- Loss: {results.get('loss', 'N/A'):.4f}

## Spatial Prediction Performance
- Mean Distance Error: {results.get('mean_distance', 'N/A'):.3f}
- RMSE: {results.get('rmse', 'N/A'):.3f}
- Mean Directional Similarity: {results.get('mean_directional_similarity', 'N/A'):.3f}

## Detailed Metrics
"""
    
    # Add detailed metrics table
    if 'detailed_metrics' in results:
        report_content += "\n### Performance by Coordinate\n"
        report_content += "| Coordinate | MAE | RMSE | R² Score |\n"
        report_content += "|------------|-----|------|----------|\n"
        
        for coord, metrics in results['detailed_metrics'].items():
            report_content += f"| {coord} | {metrics['mae']:.3f} | {metrics['rmse']:.3f} | {metrics['r2']:.3f} |\n"
    
    # Save report
    with open(report_dir / "evaluation_report.md", 'w') as f:
        f.write(report_content)
    
    # Save plots if requested
    if include_plots and 'predictions' in results:
        # Plot spatial predictions
        plot_spatial_predictions(
            results['ground_truth'],
            results['predictions'],
            save_path=str(report_dir / "spatial_predictions.png")
        )
        
        # Plot error distribution
        plot_error_distribution(
            results['ground_truth'],
            results['predictions'],
            save_path=str(report_dir / "error_distribution.png")
        )
    
    print(f"Evaluation report saved to {report_dir}")

def plot_error_distribution(true_coords: np.ndarray,
                           pred_coords: np.ndarray,
                           save_path: Optional[str] = None):
    """
    Plot detailed error distribution analysis.
    
    Args:
        true_coords: Ground truth coordinates
        pred_coords: Predicted coordinates
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Calculate errors
    errors = true_coords - pred_coords
    euclidean_errors = np.linalg.norm(errors, axis=1)
    
    # Error histogram
    ax = axes[0, 0]
    ax.hist(euclidean_errors, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(euclidean_errors), color='red', linestyle='--', 
              label=f'Mean: {np.mean(euclidean_errors):.3f}')
    ax.set_xlabel('Euclidean Distance Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Q-Q plot
    ax = axes[0, 1]
    from scipy import stats
    stats.probplot(euclidean_errors, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot')
    ax.grid(True, alpha=0.3)
    
    # Error by dimension
    ax = axes[1, 0]
    dim_labels = ['X', 'Y', 'Z']
    dim_errors = [errors[:, i] for i in range(3)]
    ax.boxplot(dim_errors, labels=dim_labels)
    ax.set_ylabel('Error')
    ax.set_title('Error Distribution by Dimension')
    ax.grid(True, alpha=0.3)
    
    # Cumulative error distribution
    ax = axes[1, 1]
    sorted_errors = np.sort(euclidean_errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    ax.plot(sorted_errors, cumulative)
    ax.set_xlabel('Euclidean Distance Error')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Error Distribution')
    ax.grid(True, alpha=0.3)
    
    # Add percentile lines
    percentiles = [50, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(euclidean_errors, p)
        ax.axvline(val, color='red', linestyle=':', alpha=0.5)
        ax.text(val, 0.5, f'{p}%: {val:.3f}', rotation=90, va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
