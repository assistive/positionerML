"""
Dataset utilities for loading and preparing datasets for spatialLM training and fine-tuning.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Union, Tuple
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets

logger = logging.getLogger(__name__)

def load_and_prepare_dataset(
    tokenizer,
    dataset_name: Optional[str] = None,
    dataset_config_name: Optional[str] = None,
    train_file: Optional[str] = None,
    validation_file: Optional[str] = None,
    text_column: str = "text",
    train_split: str = "train",
    validation_split: str = "validation",
    test_split: Optional[str] = "test",
    max_seq_length: int = 1024,
    validation_split_percentage: Optional[int] = 10,
    preprocessing_num_workers: Optional[int] = None,
    overwrite_cache: bool = False,
) -> DatasetDict:
    """
    Load and prepare a dataset for training and evaluation.
    
    Args:
        tokenizer: Tokenizer to use for encoding the text
        dataset_name: Name of the dataset to load from the Hugging Face Hub
        dataset_config_name: Configuration name for the dataset
        train_file: Path to training data file (csv, json, txt, or jsonl)
        validation_file: Path to validation data file (csv, json, txt, or jsonl)
        text_column: Column in the dataset containing the text
        train_split: Dataset split to use for training
        validation_split: Dataset split to use for validation
        test_split: Dataset split to use for testing
        max_seq_length: Maximum sequence length for tokenization
        validation_split_percentage: Percentage of training data to use for validation if no validation set is provided
        preprocessing_num_workers: Number of workers to use for preprocessing
        overwrite_cache: Whether to overwrite the cached preprocessed datasets
        
    Returns:
        A DatasetDict containing the preprocessed datasets
    """
    # Load dataset
    data_files = {}
    dataset_args = {}
    
    if train_file is not None:
        data_files["train"] = train_file
    if validation_file is not None:
        data_files["validation"] = validation_file
    
    # Load from local files if provided
    if data_files:
        extension = train_file.split(".")[-1] if train_file is not None else validation_file.split(".")[-1]
        
        if extension == "txt":
            extension = "text"
        
        dataset = load_dataset(extension, data_files=data_files, **dataset_args)
    
    # Load from Hugging Face Hub if dataset_name is provided
    elif dataset_name is not None:
        dataset = load_dataset(dataset_name, dataset_config_name, **dataset_args)
    
    else:
        raise ValueError("Either dataset_name or train_file/validation_file must be provided")
    
    # Check if the dataset has the expected splits
    if train_split not in dataset:
        raise ValueError(f"Dataset does not have a {train_split} split")
    
    # Create validation split if needed
    if validation_split not in dataset and validation_split_percentage is not None:
        logger.info(f"Creating validation split from {train_split} with {validation_split_percentage}% of data")
        
        # Split the training set to create a validation set
        split_dataset = dataset[train_split].train_test_split(
            test_size=validation_split_percentage / 100,
            seed=42
        )
        
        # Rename the splits
        dataset[train_split] = split_dataset["train"]
        dataset[validation_split] = split_dataset["test"]
    
    # Process and tokenize the dataset
    def tokenize_function(examples):
        # Tokenize the texts
        return tokenizer(examples[text_column], padding=False, truncation=True, max_length=max_seq_length)
    
    # Apply tokenization to each split
    tokenized_dataset = DatasetDict()
    for split in dataset:
        # Check if the text column exists in the dataset
        if text_column not in dataset[split].column_names:
            raise ValueError(f"Text column '{text_column}' not found in the dataset {split} split. "
                            f"Available columns: {dataset[split].column_names}")
        
        # Tokenize the dataset
        tokenized_split = dataset[split].map(
            tokenize_function,
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=[col for col in dataset[split].column_names if col != text_column],
            load_from_cache_file=not overwrite_cache,
            desc=f"Tokenizing {split} split",
        )
        
        tokenized_dataset[split] = tokenized_split
    
    # Group and chunk the data for language modeling
    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        
        # Compute length of concatenated texts
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # We drop the small remainder, and if the total_length < max_seq_length, we keep it as is
        total_length = (total_length // max_seq_length) * max_seq_length
        
        # Split by chunks of max_seq_length
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        
        # Create a new labels column
        result["labels"] = result["input_ids"].copy()
        
        return result
    
    # Apply grouping to each split
    lm_dataset = DatasetDict()
    for split in tokenized_dataset:
        lm_split = tokenized_dataset[split].map(
            group_texts,
            batched=True,
            num_proc=preprocessing_num_workers,
            load_from_cache_file=not overwrite_cache,
            desc=f"Grouping {split} split",
        )
        
        lm_dataset[split] = lm_split
    
    return lm_dataset

def load_spatial_dataset(
    tokenizer,
    spatial_file: str,
    coordinate_columns: List[str] = ["x", "y", "z"],
    text_column: str = "text",
    max_seq_length: int = 1024,
    validation_split_percentage: int = 10,
    test_split_percentage: Optional[int] = None,
    preprocessing_num_workers: Optional[int] = None,
    overwrite_cache: bool = False,
) -> DatasetDict:
    """
    Load and prepare a spatial dataset with coordinate information.
    
    Args:
        tokenizer: Tokenizer to use for encoding the text
        spatial_file: Path to spatial data file (csv or jsonl)
        coordinate_columns: List of column names containing spatial coordinates
        text_column: Column in the dataset containing the text
        max_seq_length: Maximum sequence length for tokenization
        validation_split_percentage: Percentage of data to use for validation
        test_split_percentage: Percentage of data to use for testing
        preprocessing_num_workers: Number of workers to use for preprocessing
        overwrite_cache: Whether to overwrite the cached preprocessed datasets
        
    Returns:
        A DatasetDict containing the preprocessed datasets
    """
    # Load dataset
    extension = spatial_file.split(".")[-1]
    dataset = load_dataset(extension, data_files={"full": spatial_file})["full"]
    
    # Check if the dataset has the required columns
    required_columns = [text_column] + coordinate_columns
    missing_columns = [col for col in required_columns if col not in dataset.column_names]
    
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}. "
                        f"Available columns: {dataset.column_names}")
    
    # Split the dataset into train, validation, and optionally test
    split_percentages = {"train": 100 - validation_split_percentage - (test_split_percentage or 0)}
    split_percentages["validation"] = validation_split_percentage
    
    if test_split_percentage:
        split_percentages["test"] = test_split_percentage
    
    splits = dataset.train_test_split(
        test_size=(validation_split_percentage + (test_split_percentage or 0)) / 100,
        seed=42
    )
    
    dataset_dict = DatasetDict({"train": splits["train"]})
    
    if test_split_percentage:
        # Further split the test portion into validation and test
        test_validation_split = splits["test"].train_test_split(
            test_size=test_split_percentage / (validation_split_percentage + test_split_percentage),
            seed=42
        )
        dataset_dict["validation"] = test_validation_split["train"]
        dataset_dict["test"] = test_validation_split["test"]
    else:
        dataset_dict["validation"] = splits["test"]
    
    # Process and tokenize the dataset
    def tokenize_and_add_coordinates(examples):
        # Tokenize the texts
        tokenized = tokenizer(
            examples[text_column],
            padding=False,
            truncation=True,
            max_length=max_seq_length
        )
        
        # Add coordinate information
        for coord in coordinate_columns:
            tokenized[f"coord_{coord}"] = examples[coord]
        
        return tokenized
    
    # Apply tokenization to each split
    tokenized_dataset = DatasetDict()
    for split in dataset_dict:
        # Tokenize the dataset
        tokenized_split = dataset_dict[split].map(
            tokenize_and_add_coordinates,
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=[col for col in dataset_dict[split].column_names if col != text_column and col not in coordinate_columns],
            load_from_cache_file=not overwrite_cache,
            desc=f"Tokenizing {split} split",
        )
        
        tokenized_dataset[split] = tokenized_split
    
    return tokenized_dataset

def create_augmented_spatial_dataset(
    dataset: Dataset,
    augmentation_factor: int = 2,
    noise_std: float = 0.1,
    coordinate_columns: List[str] = ["coord_x", "coord_y", "coord_z"],
    seed: int = 42,
) -> Dataset:
    """
    Create an augmented version of a spatial dataset by adding noise to coordinates.
    
    Args:
        dataset: The original dataset
        augmentation_factor: How many augmented copies to create for each original example
        noise_std: Standard deviation of the Gaussian noise to add to coordinates
        coordinate_columns: Column names containing spatial coordinates
        seed: Random seed for reproducibility
        
    Returns:
        Augmented dataset
    """
    np.random.seed(seed)
    
    # Create augmented copies
    augmented_datasets = [dataset]
    
    for i in range(augmentation_factor - 1):
        # Create a copy of the dataset
        augmented_copy = dataset.map(
            lambda examples: {
                col: examples[col] + np.random.normal(0, noise_std, len(examples[col])) 
                if col in coordinate_columns else examples[col]
                for col in examples
            },
            batched=True,
            desc=f"Creating augmented copy {i+1}/{augmentation_factor-1}",
        )
        
        augmented_datasets.append(augmented_copy)
    
    # Combine all datasets
    augmented_dataset = concatenate_datasets(augmented_datasets)
    
    return augmented_dataset
