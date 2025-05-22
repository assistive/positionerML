"""
SpatialLM model implementation.

This module defines the SpatialLM model architecture, which extends a transformer-based
language model with spatial understanding capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass
from transformers import (
    PreTrainedModel, 
    PretrainedConfig,
    AutoModel, 
    AutoConfig,
    AutoModelForCausalLM
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions
)

@dataclass
class SpatialLMConfig(PretrainedConfig):
    """
    Configuration class for SpatialLM model.
    """
    base_model_type: str = "gpt2"  # Base model type (gpt2, llama, etc.)
    base_model_name: str = "gpt2"  # Base model name (gpt2, gpt2-medium, etc.)
    spatial_dim: int = 3  # Number of spatial dimensions (typically 3 for x, y, z)
    spatial_hidden_size: int = 256  # Size of spatial hidden layers
    spatial_dropout: float = 0.1  # Dropout rate for spatial layers
    spatial_norm_eps: float = 1e-12  # Epsilon for layer normalization
    use_spatial_embeddings: bool = True  # Whether to use spatial embeddings
    spatial_embedding_size: int = 64  # Size of spatial embeddings
    use_spatial_attention: bool = True  # Whether to use spatial attention
    spatial_attention_heads: int = 4  # Number of spatial attention heads
    spatial_context_size: int = 16  # Size of spatial context window
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Update config with any kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

class SpatialEmbedding(nn.Module):
    """
    Embedding layer for spatial coordinates.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.spatial_dim = config.spatial_dim
        self.embedding_size = config.spatial_embedding_size
        
        # Coordinate embedding
        self.coord_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, self.embedding_size // self.spatial_dim),
                nn.GELU()
            ) for _ in range(self.spatial_dim)
        ])
        
        # Final projection
        self.output_projection = nn.Linear(
            self.embedding_size, 
            config.hidden_size
        )
        
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.spatial_norm_eps)
        self.dropout = nn.Dropout(config.spatial_dropout)
    
    def forward(self, coordinates):
        """
        Forward pass for spatial embeddings.
        
        Args:
            coordinates: Tensor of shape [batch_size, spatial_dim]
        
        Returns:
            Spatial embeddings of shape [batch_size, hidden_size]
        """
        # Process each coordinate dimension separately
        coord_embeddings = []
        
        for i in range(self.spatial_dim):
            # Extract the i-th coordinate dimension
            coord = coordinates[:, i:i+1]  # Shape: [batch_size, 1]
            
            # Apply the projection
            embedding = self.coord_projections[i](coord)
            coord_embeddings.append(embedding)
        
        # Concatenate embeddings from all dimensions
        combined_embedding = torch.cat(coord_embeddings, dim=-1)
        
        # Project to model hidden size
        output = self.output_projection(combined_embedding)
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        return output

class SpatialHead(nn.Module):
    """
    Head for predicting spatial coordinates from language representations.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Hidden layers
        self.dense = nn.Linear(config.hidden_size, config.spatial_hidden_size)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.spatial_hidden_size, eps=config.spatial_norm_eps)
        self.dropout = nn.Dropout(config.spatial_dropout)
        
        # Output layer for coordinate prediction
        self.output = nn.Linear(config.spatial_hidden_size, config.spatial_dim)
    
    def forward(self, hidden_states):
        """
        Forward pass for spatial prediction head.
        
        Args:
            hidden_states: Last hidden states from the language model
        
        Returns:
            Predicted spatial coordinates
        """
        # Apply transformation layers
        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        # Predict coordinates
        coordinates = self.output(x)
        
        return coordinates

class SpatialLM(PreTrainedModel):
    """
    SpatialLM model that extends a causal language model with spatial understanding capabilities.
    """
    config_class = SpatialLMConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # Load base language model
        base_config = AutoConfig.from_pretrained(config.base_model_name)
        self.language_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            config=base_config
        )
        
        # Spatial components
        if config.use_spatial_embeddings:
            self.spatial_embedding = SpatialEmbedding(config)
        
        if config.use_spatial_attention:
            self.spatial_attention = SpatialAttention(config)
        
        # Spatial prediction head
        self.spatial_head = SpatialHead(config)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights for spatial components"""
        # The base model's weights are already initialized
        # Initialize only the new components
        if hasattr(self, "spatial_embedding"):
            self.apply(self._init_weights)
        if hasattr(self, "spatial_attention"):
            self.apply(self._init_weights)
        if hasattr(self, "spatial_head"):
            self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Initialize linear layers with small random weights
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            # Initialize LayerNorm with ones and zeros
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        spatial_coordinates=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        """
        Forward pass for the SpatialLM model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            position_ids: Position IDs
            spatial_coordinates: Spatial coordinates tensor of shape [batch_size, spatial_dim]
            head_mask: Head mask for transformer
            inputs_embeds: Input embeddings
            labels: Labels for language modeling
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dict or tuple
            
        Returns:
            Model outputs including language modeling and spatial predictions
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Process language inputs
        lm_outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=True,  # We need hidden states for spatial processing
            return_dict=True,
        )
        
        # Extract hidden states
        sequence_output = lm_outputs.hidden_states[-1]
        
        # Process spatial information if provided
        if spatial_coordinates is not None and hasattr(self, "spatial_embedding"):
            spatial_embeds = self.spatial_embedding(spatial_coordinates)
            
            # Apply spatial attention if enabled
            if hasattr(self, "spatial_attention"):
                sequence_output = self.spatial_attention(
                    sequence_output, 
                    spatial_embeds,
                    attention_mask=attention_mask
                )
        
        # Predict spatial coordinates from language representations
        # Use the [CLS] token representation (first token) or last token for prediction
        pooled_output = sequence_output[:, 0, :]
        spatial_predictions = self.spatial_head(pooled_output)
        
        # Prepare outputs
        outputs = (lm_outputs.loss, spatial_predictions)
        
        if not return_dict:
            return outputs
        
        # Create custom output dataclass
        return SpatialLMOutput(
            loss=lm_outputs.loss,
            spatial_predictions=spatial_predictions,
            logits=lm_outputs.logits,
            hidden_states=lm_outputs.hidden_states if output_hidden_states else None,
            attentions=lm_outputs.attentions if output_attentions else None,
        )
    
    def predict_spatial(self, text, tokenizer, device=None):
        """
        Predict spatial coordinates from text.
        
        Args:
            text: Input text
            tokenizer: Tokenizer for encoding text
            device: Device to run prediction on
            
        Returns:
            Predicted spatial coordinates
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.forward(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
        
        # Return spatial predictions
        return outputs.spatial_predictions.cpu().numpy()

@dataclass
class SpatialLMOutput:
    """
    Output class for SpatialLM model.
    """
    loss: Optional[torch.FloatTensor] = None
    spatial_predictions: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
