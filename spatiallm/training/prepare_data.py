#!/usr/bin/env python3
"""
Prepare data for spatialLM training and evaluation.

This script processes raw data files into the format required for spatialLM
training. It can handle various input formats (CSV, JSON, JSONL) and perform
preprocessing steps such as tokenization, augmentation, and validation split.
"""

import os
import sys
import logging
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DataConfig, SpatialLMConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("prepare_data")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Prepare data for spatialLM training")
    
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input data file (CSV, JSON, or JSONL)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./processed_data",
        help="Directory to save the processed data"
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
        "--split",
        action="store_true",
        help="Whether to split the data into train/validation/test sets"
    )
    
    parser.add_argument(
        "--validation_size",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation (if split is True)"
    )
    
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Fraction of data to use for testing (if split is True)"
    )
    
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Whether to perform data augmentation"
    )
    
    parser.add_argument(
        "--augmentation_factor",
        type=int,
        default=2,
        help="Factor by which to augment the data (if augment is True)"
    )
    
    parser.add_argument(
        "--noise_level",
        type=float,
        default=0.1,
        help="Standard deviation of noise for coordinate augmentation"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of samples to use (for debugging or testing)"
    )
    
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Whether to perform data cleaning (remove duplicates, etc.)"
    )
    
    parser.add_argument(
        "--min_text_length",
        type=int,
        default=10,
        help="Minimum text length to keep a sample"
    )
    
    parser.add_argument(
        "--max_text_length",
        type=int,
        default=None,
        help="Maximum text length to keep a sample"
    )
    
    parser.add_argument(
        "--normalize_coordinates",
        action="store_true",
        help="Whether to normalize spatial coordinates"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a configuration file (will override other arguments)"
    )
    
    return parser.parse_args()

def load_data(input_file):
    """
    Load data from various file formats
    
    Args:
        input_file: Path to the input file
    
    Returns:
        DataFrame containing the data
    """
    logger.info(f"Loading data from {input_file}")
    
    # Determine file type based on extension
    file_extension = os.path.splitext(input_file)[1].lower()
    
    if file_extension == '.csv':
        df = pd.read_csv(input_file)
    elif file_extension == '.json':
        # Try to load as a JSON array
        with open(input_file, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    elif file_extension == '.jsonl':
        # Load as JSON lines
        data = []
        with open(input_file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        df = pd.DataFrame(data)
    elif file_extension == '.tsv':
        df = pd.read_csv(input_file, sep='\t')
    elif file_extension == '.xlsx':
        df = pd.read_excel(input_file)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    logger.info(f"Loaded {len(df)} rows")
    return df

def clean_data(df, text_column, coordinate_columns, min_text_length=10, max_text_length=None):
    """
    Clean the data by removing duplicates, invalid entries, etc.
    
    Args:
        df: Input DataFrame
        text_column: Name of the text column
        coordinate_columns: List of coordinate column names
        min_text_length: Minimum text length to keep a sample
        max_text_length: Maximum text length to keep a sample
    
    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning data")
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Check for required columns
    required_columns = [text_column] + coordinate_columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Remove duplicates
    original_len = len(df)
    df.drop_duplicates(inplace=True)
    logger.info(f"Removed {original_len - len(df)} duplicate rows")
    
    # Filter by text length
    original_len = len(df)
    df = df[df[text_column].str.len() >= min_text_length]
    logger.info(f"Removed {original_len - len(df)} rows with text length < {min_text_length}")
    
    if max_text_length:
        original_len = len(df)
        df = df[df[text_column].str.len() <= max_text_length]
        logger.info(f"Removed {original_len - len(df)} rows with text length > {max_text_length}")
    
    # Remove rows with missing coordinates
    original_len = len(df)
    df = df.dropna(subset=coordinate_columns)
    logger.info(f"Removed {original_len - len(df)} rows with missing coordinates")
    
    # Remove rows with invalid coordinates (non-numeric)
    for col in coordinate_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    original_len = len(df)
    df = df.dropna(subset=coordinate_columns)
    logger.info(f"Removed {original_len - len(df)} rows with invalid coordinates")
    
    return df

def normalize_coordinates(df, coordinate_columns):
    """
    Normalize spatial coordinates to have zero mean and unit variance
    
    Args:
        df: Input DataFrame
        coordinate_columns: List of coordinate column names
    
    Returns:
        DataFrame with normalized coordinates
    """
    logger.info("Normalizing coordinates")
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Calculate mean and standard deviation for each coordinate
    stats = {}
    for col in coordinate_columns:
        mean = df[col].mean()
        std = df[col].std()
        stats[col] = {'mean': mean, 'std': std}
        
        # Normalize
        df[col] = (df[col] - mean) / std
    
    # Save normalization stats
    df.attrs['coordinate_stats'] = stats
    
    return df

def augment_data(df, text_column, coordinate_columns, augmentation_factor=2, noise_level=0.1, seed=42):
    """
    Augment the data by adding noise to coordinates
    
    Args:
        df: Input DataFrame
        text_column: Name of the text column
        coordinate_columns: List of coordinate column names
        augmentation_factor: Factor by which to augment the data
        noise_level: Standard deviation of the noise to add
        seed: Random seed
    
    Returns:
        Augmented DataFrame
    """
    logger.info(f"Augmenting data by factor {augmentation_factor}")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Make a copy of the original data
    original_df = df.copy()
    
    # Create augmented copies
    augmented_dfs = [original_df]
    
    for i in range(augmentation_factor - 1):
        logger.info(f"Creating augmented copy {i+1}/{augmentation_factor-1}")
        
        # Create a copy
        augmented_df = original_df.copy()
        
        # Add noise to coordinates
        for col in coordinate_columns:
            noise = np.random.normal(0, noise_level, size=len(augmented_df))
            augmented_df[col] = augmented_df[col] + noise
        
        # Add a suffix to the text to indicate it's augmented
        # This is optional and can be removed if not needed
        augmented_df[text_column] = augmented_df[text_column] + f" [Augmented {i+1}]"
        
        augmented_dfs.append(augmented_df)
    
    # Combine all augmented copies
    combined_df = pd.concat(augmented_dfs, ignore_index=True)
    
    logger.info(f"Augmented data size: {len(combined_df)} rows")
    return combined_df

def split_data(df, validation_size=0.1, test_size=0.1, seed=42):
    """
    Split the data into train, validation, and test sets
    
    Args:
        df: Input DataFrame
        validation_size: Fraction of data to use for validation
        test_size: Fraction of data to use for testing
        seed: Random seed
    
    Returns:
        Dictionary containing train, validation, and test DataFrames
    """
    logger.info("Splitting data into train/validation/test sets")
    
    # Calculate test and validation sizes
    total_test_val_size = validation_size + test_size
    test_ratio = test_size / total_test_val_size if total_test_val_size > 0 else 0
    
    # Split into train and temp sets
    train_df, temp_df = train_test_split(
        df, 
        test_size=total_test_val_size,
        random_state=seed
    )
    
    # Split temp into validation and test sets
    if test_size > 0 and validation_size > 0:
        val_df, test_df = train_test_split(
            temp_df, 
            test_size=test_ratio,
            random_state=seed
        )
    elif validation_size > 0:
        val_df = temp_df
        test_df = None
    elif test_size > 0:
        test_df = temp_df
        val_df = None
    else:
        val_df = None
        test_df = None
    
    # Create result dictionary
    result = {'train': train_df}
    if val_df is not None:
        result['validation'] = val_df
    if test_df is not None:
        result['test'] = test_df
    
    # Log split sizes
    logger.info(f"Train set size: {len(train_df)} rows")
    if val_df is not None:
        logger.info(f"Validation set size: {len(val_df)} rows")
    if test_df is not None:
        logger.info(f"Test set size: {len(test_df)} rows")
    
    return result

def save_data(data_dict, output_dir, prefix=""):
    """
    Save data to output files
    
    Args:
        data_dict: Dictionary of DataFrames to save
        output_dir: Directory to save the files
        prefix: Prefix for the output files
    """
    logger.info(f"Saving data to {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each split
    for split_name, df in data_dict.items():
        output_file = os.path.join(output_dir, f"{prefix}{split_name}.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {split_name} set to {output_file}")
    
    # Save normalization stats if they exist
    if hasattr(next(iter(data_dict.values())), 'attrs') and 'coordinate_stats' in next(iter(data_dict.values())).attrs:
        stats_file = os.path.join(output_dir, f"{prefix}coordinate_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(next(iter(data_dict.values())).attrs['coordinate_stats'], f, indent=2)
        logger.info(f"Saved coordinate normalization stats to {stats_file}")

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Load configuration if provided
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = SpatialLMConfig.load(args.config)
        data_config = config.data
    else:
        # Create data config from arguments
        coordinate_columns = args.coordinate_columns.split(',')
        data_config = DataConfig(
            train_file=args.input_file,
            text_column=args.text_column,
            coordinate_columns=coordinate_columns,
            data_augmentation=args.augment,
            augmentation_factor=args.augmentation_factor,
            validation_split_percentage=int(args.validation_size * 100),
        )
    
    # Load data
    df = load_data(args.input_file)
    
    # Sample data if requested
    if args.sample:
        logger.info(f"Sampling {args.sample} rows")
        df = df.sample(min(args.sample, len(df)), random_state=args.seed)
    
    # Clean data if requested
    if args.clean:
        df = clean_data(
            df,
            data_config.text_column,
            data_config.coordinate_columns,
            min_text_length=args.min_text_length,
            max_text_length=args.max_text_length
        )
    
    # Normalize coordinates if requested
    if args.normalize_coordinates:
        df = normalize_coordinates(df, data_config.coordinate_columns)
    
    # Augment data if requested
    if args.augment or data_config.data_augmentation:
        augmentation_factor = args.augmentation_factor if args.augment else data_config.augmentation_factor
        df = augment_data(
            df,
            data_config.text_column,
            data_config.coordinate_columns,
            augmentation_factor=augmentation_factor,
            noise_level=args.noise_level,
            seed=args.seed
        )
    
    # Split data if requested
    if args.split:
        data_dict = split_data(
            df,
            validation_size=args.validation_size,
            test_size=args.test_size,
            seed=args.seed
        )
    else:
        data_dict = {'all': df}
    
    # Save data
    save_data(data_dict, args.output_dir)
    
    logger.info("Data preparation completed!")

if __name__ == "__main__":
    main()
