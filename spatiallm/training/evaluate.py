#!/usr/bin/env python3
"""
Evaluate spatialLM models on test data.

This script evaluates the performance of trained spatialLM models on test data,
computing metrics for both language modeling and spatial prediction tasks.
"""

import os
import sys
import logging
import argparse
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from models.spatialLM import SpatialLM, SpatialLMConfig
from utils.metrics import (
    compute_metrics, 
    compute_spatial_metrics, 
    evaluate_spatial_language_understanding,
    calculate_perplexity
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("evaluate")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate a spatialLM model")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model"
    )
    
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Path to the test data file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )
    
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Column containing the text data"
    )
    
    parser.add_argument(
        "--coordinate_columns",
        type=str,
        default="x,y,z",
        help="Comma-separated list of columns containing spatial coordinates"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for debugging or testing)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for evaluation (e.g., 'cuda', 'cpu')"
    )
    
    parser.add_argument(
        "--generate_text",
        action="store_true",
        help="Whether to generate text for qualitative evaluation"
    )
    
    parser.add_argument(
        "--num_generate",
        type=int,
        default=5,
        help="Number of examples to generate text for"
    )
    
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Whether to save model predictions"
    )
    
    parser.add_argument(
        "--plot_results",
        action="store_true",
        help="Whether to create plots for the evaluation results"
    )
    
    return parser.parse_args()

def load_model_and_tokenizer(model_path, device=None):
    """
    Load the model and tokenizer from the given path
    
    Args:
        model_path: Path to the model
        device: Device to load the model on
    
    Returns:
        model: The loaded model
        tokenizer: The tokenizer
    """
    logger.info(f"Loading model and tokenizer from {model_path}")
    
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model = SpatialLM.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer, device

def load_test_data(test_file, text_column, coordinate_columns, max_samples=None):
    """
    Load test data from file
    
    Args:
        test_file: Path to the test data file
        text_column: Name of the text column
        coordinate_columns: List of coordinate column names
        max_samples: Maximum number of samples to load
    
    Returns:
        DataFrame containing the test data
    """
    logger.info(f"Loading test data from {test_file}")
    
    # Determine file type based on extension
    file_extension = os.path.splitext(test_file)[1].lower()
    
    if file_extension == '.csv':
        df = pd.read_csv(test_file)
    elif file_extension == '.json':
        with open(test_file, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    elif file_extension == '.jsonl':
        data = []
        with open(test_file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        df = pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    # Check for required columns
    required_columns = [text_column] + coordinate_columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Sample data if requested
    if max_samples is not None:
        df = df.sample(min(max_samples, len(df)), random_state=42)
    
    logger.info(f"Loaded {len(df)} test samples")
    return df

def evaluate_language_modeling(model, tokenizer, test_df, text_column, batch_size=8, device="cuda"):
    """
    Evaluate the language modeling performance of the model
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        test_df: DataFrame containing test data
        text_column: Name of the text column
        batch_size: Batch size for evaluation
        device: Device to use for evaluation
    
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating language modeling performance")
    
    # Calculate perplexity for each text
    perplexities = []
    
    for i in tqdm(range(0, len(test_df), batch_size), desc="Calculating perplexity"):
        batch_texts = test_df[text_column].iloc[i:i+batch_size].tolist()
        
        for text in batch_texts:
            try:
                ppl = calculate_perplexity(model, tokenizer, text)
                perplexities.append(ppl)
            except Exception as e:
                logger.warning(f"Error calculating perplexity for text: {e}")
    
    # Calculate average perplexity
    avg_perplexity = np.mean(perplexities)
    
    # Return metrics
    metrics = {
        "perplexity": avg_perplexity,
        "perplexity_std": np.std(perplexities),
        "perplexity_min": np.min(perplexities),
        "perplexity_max": np.max(perplexities),
    }
    
    logger.info(f"Average perplexity: {avg_perplexity:.2f}")
    
    return metrics

def evaluate_spatial_prediction(model, tokenizer, test_df, text_column, coordinate_columns, batch_size=8, device="cuda"):
    """
    Evaluate the spatial prediction performance of the model
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        test_df: DataFrame containing test data
        text_column: Name of the text column
        coordinate_columns: List of coordinate column names
        batch_size: Batch size for evaluation
        device: Device to use for evaluation
    
    Returns:
        Dictionary of evaluation metrics and predictions
    """
    logger.info("Evaluating spatial prediction performance")
    
    # Extract true coordinates
    true_coordinates = test_df[coordinate_columns].values
    
    # Make predictions
    predicted_coordinates = []
    
    for i in tqdm(range(0, len(test_df), batch_size), desc="Predicting coordinates"):
        batch_texts = test_df[text_column].iloc[i:i+batch_size].tolist()
        
        for text in batch_texts:
            try:
                # Predict spatial coordinates
                with torch.no_grad():
                    coords = model.predict_spatial(text, tokenizer, device=device)
                predicted_coordinates.append(coords)
            except Exception as e:
                logger.warning(f"Error predicting coordinates for text: {e}")
                # Append zeros as fallback
                predicted_coordinates.append(np.zeros(len(coordinate_columns)))
    
    # Convert to numpy array
    predicted_coordinates = np.array(predicted_coordinates)
    
    # Calculate metrics
    metrics = compute_spatial_metrics(predicted_coordinates, true_coordinates)
    
    logger.info(f"Mean distance error: {metrics['mean_distance']:.2f}")
    logger.info(f"RMSE: {metrics['rmse']:.2f}")
    
    return {
        "metrics": metrics,
        "predictions": predicted_coordinates,
        "ground_truth": true_coordinates
    }

def generate_text_examples(model, tokenizer, test_df, text_column, coordinate_columns, num_examples=5, device="cuda"):
    """
    Generate text examples for qualitative evaluation
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        test_df: DataFrame containing test data
        text_column: Name of the text column
        coordinate_columns: List of coordinate column names
        num_examples: Number of examples to generate
        device: Device to use for evaluation
    
    Returns:
        List of generated examples
    """
    logger.info(f"Generating {num_examples} text examples")
    
    # Sample examples
    sample_indices = np.random.choice(len(test_df), min(num_examples, len(test_df)), replace=False)
    examples = []
    
    for idx in sample_indices:
        # Get sample
        text = test_df[text_column].iloc[idx]
        coords = test_df[coordinate_columns].iloc[idx].values
        
        # Create prompt for generation
        prompt = f"Text: {text}\nCoordinates: {', '.join(map(str, coords))}\nDescription: "
        
        # Generate text
        try:
            with torch.no_grad():
                generated_text = model.generate(
                    input_ids=tokenizer.encode(prompt, return_tensors="pt").to(device),
                    max_length=100,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id
                )
                generated_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
        except Exception as e:
            logger.warning(f"Error generating text: {e}")
            generated_text = "Error generating text"
        
        # Predict coordinates
        try:
            with torch.no_grad():
                predicted_coords = model.predict_spatial(text, tokenizer, device=device)
        except Exception as e:
            logger.warning(f"Error predicting coordinates: {e}")
            predicted_coords = np.zeros(len(coordinate_columns))
        
        # Create example
        example = {
            "text": text,
            "true_coordinates": coords.tolist(),
            "predicted_coordinates": predicted_coords.tolist(),
            "generated_text": generated_text
        }
        
        examples.append(example)
    
    return examples

def plot_coordinate_predictions(true_coords, pred_coords, coordinate_columns, output_path):
    """
    Plot coordinate predictions vs. ground truth
    
    Args:
        true_coords: Ground truth coordinates
        pred_coords: Predicted coordinates
        coordinate_columns: List of coordinate column names
        output_path: Path to save the plot
    """
    num_coords = true_coords.shape[1]
    
    plt.figure(figsize=(15, 5 * num_coords))
    
    for i in range(num_coords):
        plt.subplot(num_coords, 1, i + 1)
        
        # Create scatter plot
        plt.scatter(true_coords[:, i], pred_coords[:, i], alpha=0.5)
        
        # Add diagonal line (perfect predictions)
        min_val = min(true_coords[:, i].min(), pred_coords[:, i].min())
        max_val = max(true_coords[:, i].max(), pred_coords[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # Add labels
        plt.xlabel(f"True {coordinate_columns[i]}")
        plt.ylabel(f"Predicted {coordinate_columns[i]}")
        plt.title(f"{coordinate_columns[i]} Predictions")
        
        # Add correlation coefficient
        corr = np.corrcoef(true_coords[:, i], pred_coords[:, i])[0, 1]
        plt.annotate(f"Correlation: {corr:.2f}", xy=(0.05, 0.95), xycoords='axes fraction')
    
    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"Coordinate prediction plot saved to {output_path}")

def plot_error_distribution(spatial_results, output_path):
    """
    Plot the distribution of spatial prediction errors
    
    Args:
        spatial_results: Results from evaluate_spatial_prediction
        output_path: Path to save the plot
    """
    # Calculate Euclidean distances
    true_coords = spatial_results["ground_truth"]
    pred_coords = spatial_results["predictions"]
    
    errors = np.sqrt(np.sum((true_coords - pred_coords) ** 2, axis=1))
    
    plt.figure(figsize=(10, 6))
    
    # Plot histogram
    plt.hist(errors, bins=30, alpha=0.7)
    
    # Add vertical lines for statistics
    plt.axvline(np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.2f}')
    plt.axvline(np.median(errors), color='g', linestyle='--', label=f'Median: {np.median(errors):.2f}')
    
    # Add labels
    plt.xlabel("Euclidean Distance Error")
    plt.ylabel("Frequency")
    plt.title("Distribution of Spatial Prediction Errors")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"Error distribution plot saved to {output_path}")

def save_evaluation_results(lm_metrics, spatial_results, examples, output_dir):
    """
    Save evaluation results to files
    
    Args:
        lm_metrics: Language modeling metrics
        spatial_results: Spatial prediction results
        examples: Generated text examples
        output_dir: Directory to save the results
    """
    logger.info(f"Saving evaluation results to {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save language modeling metrics
    lm_path = os.path.join(output_dir, "language_modeling_metrics.json")
    with open(lm_path, 'w') as f:
        json.dump(lm_metrics, f, indent=2)
    
    # Save spatial metrics
    spatial_path = os.path.join(output_dir, "spatial_metrics.json")
    with open(spatial_path, 'w') as f:
        json.dump(spatial_results["metrics"], f, indent=2)
    
    # Save predictions if available
    if spatial_results.get("predictions") is not None:
        pred_path = os.path.join(output_dir, "predictions.npz")
        np.savez(
            pred_path,
            predictions=spatial_results["predictions"],
            ground_truth=spatial_results["ground_truth"]
        )
    
    # Save generated examples if available
    if examples:
        examples_path = os.path.join(output_dir, "generated_examples.json")
        with open(examples_path, 'w') as f:
            json.dump(examples, f, indent=2)
    
    logger.info("Evaluation results saved successfully")

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(args.model_path, args.device)
    
    # Parse coordinate columns
    coordinate_columns = args.coordinate_columns.split(',')
    
    # Load test data
    test_df = load_test_data(
        args.test_file,
        args.text_column,
        coordinate_columns,
        max_samples=args.max_samples
    )
    
    # Evaluate language modeling
    lm_metrics = evaluate_language_modeling(
        model,
        tokenizer,
        test_df,
        args.text_column,
        batch_size=args.batch_size,
        device=device
    )
    
    # Evaluate spatial prediction
    spatial_results = evaluate_spatial_prediction(
        model,
        tokenizer,
        test_df,
        args.text_column,
        coordinate_columns,
        batch_size=args.batch_size,
        device=device
    )
    
    # Generate text examples if requested
    examples = None
    if args.generate_text:
        examples = generate_text_examples(
            model,
            tokenizer,
            test_df,
            args.text_column,
            coordinate_columns,
            num_examples=args.num_generate,
            device=device
        )
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save results
    if args.save_predictions or examples:
        save_evaluation_results(
            lm_metrics,
            spatial_results,
            examples,
            args.output_dir
        )
    
    # Create plots if requested
    if args.plot_results:
        # Plot coordinate predictions
        plot_coordinate_predictions(
            spatial_results["ground_truth"],
            spatial_results["predictions"],
            coordinate_columns,
            os.path.join(args.output_dir, "coordinate_predictions.png")
        )
        
        # Plot error distribution
        plot_error_distribution(
            spatial_results,
            os.path.join(args.output_dir, "error_distribution.png")
        )
    
    # Print summary
    print("\n========== Evaluation Summary ==========")
    print(f"Language Modeling:")
    print(f"  Perplexity: {lm_metrics['perplexity']:.2f}")
    
    print(f"\nSpatial Prediction:")
    print(f"  Mean Distance Error: {spatial_results['metrics']['mean_distance']:.2f}")
    print(f"  RMSE: {spatial_results['metrics']['rmse']:.2f}")
    print(f"  Directional Similarity: {spatial_results['metrics']['mean_directional_similarity']:.2f}")
    
    logger.info("Evaluation completed!")

if __name__ == "__main__":
    main()
