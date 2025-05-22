"""
Metrics for evaluating spatialLM models during training and evaluation.
"""

import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import mean_squared_error
from transformers import EvalPrediction

def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    Compute metrics for language modeling evaluation.
    
    Args:
        eval_pred: Evaluation prediction object containing predictions and labels
        
    Returns:
        Dict of metrics
    """
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    
    # Calculate perplexity
    loss = calculate_loss(logits, labels)
    perplexity = np.exp(loss)
    
    return {
        "perplexity": perplexity,
        "loss": loss
    }

def calculate_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate cross-entropy loss for language modeling.
    
    Args:
        logits: Prediction logits
        labels: Ground truth labels
        
    Returns:
        Cross-entropy loss
    """
    # Convert labels and logits to correct format if needed
    if isinstance(logits, tuple):
        logits = logits[0]  # Take the language modeling logits
    
    # Shift logits and labels for causal language modeling
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Calculate loss
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    return loss.item()

def compute_spatial_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute metrics for spatial prediction tasks.
    
    Args:
        predictions: Predicted coordinates
        targets: Ground truth coordinates
        mask: Optional mask for valid predictions
        
    Returns:
        Dict of metrics
    """
    if mask is not None:
        predictions = predictions[mask]
        targets = targets[mask]
    
    # Calculate mean squared error
    mse = mean_squared_error(targets, predictions)
    
    # Calculate Euclidean distance error (L2 norm)
    euclidean_distances = np.sqrt(np.sum((predictions - targets) ** 2, axis=1))
    mean_distance = np.mean(euclidean_distances)
    median_distance = np.median(euclidean_distances)
    max_distance = np.max(euclidean_distances)
    
    # Calculate directional accuracy (cosine similarity)
    def cosine_similarity(a, b):
        norm_a = np.linalg.norm(a, axis=1)
        norm_b = np.linalg.norm(b, axis=1)
        
        # Handle zero vectors
        zero_mask = (norm_a == 0) | (norm_b == 0)
        if np.any(zero_mask):
            sim = np.zeros(len(a))
            valid_mask = ~zero_mask
            if np.any(valid_mask):
                dot_product = np.sum(a[valid_mask] * b[valid_mask], axis=1)
                sim[valid_mask] = dot_product / (norm_a[valid_mask] * norm_b[valid_mask])
            return sim
        else:
            dot_product = np.sum(a * b, axis=1)
            return dot_product / (norm_a * norm_b)
    
    # Calculate directional similarity for non-zero vectors
    directional_sim = cosine_similarity(predictions, targets)
    mean_directional_sim = np.mean(directional_sim)
    
    return {
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mean_distance": mean_distance,
        "median_distance": median_distance,
        "max_distance": max_distance,
        "mean_directional_similarity": mean_directional_sim
    }

def evaluate_spatial_language_understanding(
    model, 
    dataset, 
    tokenizer, 
    max_seq_length: int = 512,
    coordinate_columns: List[str] = ["x", "y", "z"],
    batch_size: int = 8
) -> Dict[str, float]:
    """
    Evaluate spatial language understanding capabilities of the model.
    
    Args:
        model: The spatialLM model
        dataset: Evaluation dataset with spatial coordinates
        tokenizer: Tokenizer for the model
        max_seq_length: Maximum sequence length for tokenization
        coordinate_columns: Names of columns containing spatial coordinates
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Prepare for evaluation
    all_predictions = []
    all_targets = []
    
    # Run evaluation in batches
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:min(i+batch_size, len(dataset))]
        
        # Tokenize inputs
        inputs = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt"
        ).to(device)
        
        # Extract target coordinates
        targets = np.array([[batch[f"{col}"][j] for col in coordinate_columns] for j in range(len(batch["text"]))])
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
            # Extract predictions from the model output
            # The exact method depends on how your model produces spatial predictions
            # This is an example assuming the model has a spatial_head that predicts coordinates
            if hasattr(model, "spatial_head"):
                predictions = model.spatial_head(outputs.hidden_states[-1][:, 0, :]).cpu().numpy()
            else:
                # Fallback: Use the last hidden state of the [CLS] token
                predictions = outputs.hidden_states[-1][:, 0, :model.config.hidden_size].cpu().numpy()
                
                # Project to coordinate dimensions if needed
                if predictions.shape[1] != len(coordinate_columns):
                    # Simple linear projection as fallback
                    predictions = predictions @ np.random.randn(predictions.shape[1], len(coordinate_columns))
        
        all_predictions.append(predictions)
        all_targets.append(targets)
    
    # Combine batches
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    
    # Compute metrics
    metrics = compute_spatial_metrics(all_predictions, all_targets)
    
    return metrics

def calculate_perplexity(model, tokenizer, text, stride=512):
    """
    Calculate perplexity of a given text.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Text to calculate perplexity for
        stride: Stride for sliding window evaluation
        
    Returns:
        Perplexity value
    """
    encodings = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    max_length = model.config.max_position_embeddings
    
    nlls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
        
        nlls.append(neg_log_likelihood)
    
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()
