# fastvlm/src/fastvlm_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union, Any
from dataclasses import dataclass
import math
from transformers import (
    PreTrainedModel, 
    PretrainedConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoConfig
)
from transformers.modeling_outputs import ModelOutput


@dataclass
class FastVLMConfig(PretrainedConfig):
    """Configuration class for FastVLM model."""
    
    model_type = "fastvlm"
    
    def __init__(
        self,
        vision_encoder: str = "efficient_vit",
        language_model: str = "opt-1.3b",
        hidden_size: int = 768,
        vision_hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_vision_tokens: int = 256,
        max_vision_tokens: int = 576,
        token_selection_method: str = "importance",
        attention_type: str = "linear",
        use_flash_attention: bool = False,
        use_cached_vision_features: bool = True,
        vision_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        tie_word_embeddings: bool = False,
        vocab_size: int = 50272,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.hidden_size = hidden_size
        self.vision_hidden_size = vision_hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_vision_tokens = num_vision_tokens
        self.max_vision_tokens = max_vision_tokens
        self.token_selection_method = token_selection_method
        self.attention_type = attention_type
        self.use_flash_attention = use_flash_attention
        self.use_cached_vision_features = use_cached_vision_features
        self.vision_dropout = vision_dropout
        self.attention_dropout = attention_dropout
        self.mlp_dropout = mlp_dropout
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.tie_word_embeddings = tie_word_embeddings
        self.vocab_size = vocab_size


@dataclass
class FastVLMOutput(ModelOutput):
    """Output class for FastVLM model."""
    
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    vision_features: Optional[torch.FloatTensor] = None
    selected_tokens: Optional[torch.LongTensor] = None
    token_importance_scores: Optional[torch.FloatTensor] = None


class EfficientVisionEncoder(nn.Module):
    """Efficient vision encoder with multi-scale processing and early exit."""
    
    def __init__(self, config: FastVLMConfig):
        super().__init__()
        self.config = config
        
        # Multi-scale feature extractors
        self.patch_embed_small = nn.Conv2d(3, config.vision_hidden_size // 4, kernel_size=4, stride=4)
        self.patch_embed_medium = nn.Conv2d(3, config.vision_hidden_size // 2, kernel_size=8, stride=8)
        self.patch_embed_large = nn.Conv2d(3, config.vision_hidden_size, kernel_size=16, stride=16)
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(config.vision_hidden_size * 7 // 4, config.vision_hidden_size),
            nn.GELU(),
            nn.Dropout(config.vision_dropout)
        )
        
        # Positional embeddings
        self.position_embeddings = nn.Parameter(
            torch.randn(1, config.max_vision_tokens, config.vision_hidden_size)
        )
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            EfficientTransformerBlock(config) for _ in range(6)
        ])
        
        self.norm = nn.LayerNorm(config.vision_hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the vision encoder.
        
        Args:
            pixel_values: Input images [batch_size, 3, height, width]
            
        Returns:
            vision_features: Encoded vision features
            importance_scores: Token importance scores for selection
        """
        batch_size = pixel_values.shape[0]
        
        # Multi-scale feature extraction
        features_small = self.patch_embed_small(pixel_values)
        features_medium = self.patch_embed_medium(pixel_values)
        features_large = self.patch_embed_large(pixel_values)
        
        # Reshape and concatenate
        features_small = features_small.flatten(2).transpose(1, 2)
        features_medium = features_medium.flatten(2).transpose(1, 2)
        features_large = features_large.flatten(2).transpose(1, 2)
        
        # Upsample smaller features to match dimensions
        features_small = F.interpolate(
            features_small.transpose(1, 2).unsqueeze(2),
            size=(1, features_large.shape[1]),
            mode='nearest'
        ).squeeze(2).transpose(1, 2)
        
        features_medium = F.interpolate(
            features_medium.transpose(1, 2).unsqueeze(2),
            size=(1, features_large.shape[1]),
            mode='nearest'
        ).squeeze(2).transpose(1, 2)
        
        # Concatenate multi-scale features
        features = torch.cat([features_small, features_medium, features_large], dim=-1)
        features = self.feature_fusion(features)
        
        # Add positional embeddings
        seq_len = features.shape[1]
        features = features + self.position_embeddings[:, :seq_len, :]
        
        # Apply transformer blocks with early exit
        importance_scores = []
        
        for i, layer in enumerate(self.layers):
            features, scores = layer(features, return_importance=True)
            importance_scores.append(scores)
            
            # Early exit if features are stable (optional)
            if i > 2 and self.config.use_cached_vision_features:
                # Check if features have converged
                if i > 0:
                    feature_change = torch.norm(features - prev_features, dim=-1).mean()
                    if feature_change < 0.01:  # Threshold for early exit
                        break
                prev_features = features.clone()
        
        features = self.norm(features)
        
        # Aggregate importance scores
        importance_scores = torch.stack(importance_scores).mean(dim=0)
        
        return features, importance_scores


class EfficientTransformerBlock(nn.Module):
    """Efficient transformer block with linear attention."""
    
    def __init__(self, config: FastVLMConfig):
        super().__init__()
        self.config = config
        
        self.attention = LinearAttention(config) if config.attention_type == "linear" else nn.MultiheadAttention(
            config.vision_hidden_size,
            config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(config.vision_hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.vision_hidden_size, eps=config.layer_norm_eps)
        
        self.mlp = nn.Sequential(
            nn.Linear(config.vision_hidden_size, config.vision_hidden_size * 4),
            nn.GELU(),
            nn.Dropout(config.mlp_dropout),
            nn.Linear(config.vision_hidden_size * 4, config.vision_hidden_size),
            nn.Dropout(config.mlp_dropout)
        )
        
    def forward(self, x: torch.Tensor, return_importance: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self-attention
        residual = x
        x = self.norm1(x)
        
        if isinstance(self.attention, LinearAttention):
            attn_output, importance = self.attention(x, return_importance=return_importance)
        else:
            attn_output, _ = self.attention(x, x, x)
            importance = None
        
        x = residual + attn_output
        
        # MLP
        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)
        
        return x, importance


class LinearAttention(nn.Module):
    """Linear complexity attention mechanism."""
    
    def __init__(self, config: FastVLMConfig):
        super().__init__()
        self.hidden_size = config.vision_hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        self.qkv = nn.Linear(self.hidden_size, self.hidden_size * 3)
        self.proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(config.attention_dropout)
        
    def forward(self, x: torch.Tensor, return_importance: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply kernel feature map
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        # Compute attention in linear time
        k_sum = k.sum(dim=2, keepdim=True)
        qk = torch.matmul(q.unsqueeze(3), k.unsqueeze(2)).sum(dim=3)
        qk_sum = torch.matmul(q, k_sum.transpose(-2, -1))
        
        # Compute weighted values
        qkv = torch.matmul(qk.unsqueeze(3), v.unsqueeze(2)).sum(dim=3)
        qkv_sum = torch.matmul(qk_sum, v.sum(dim=2, keepdim=True))
        
        # Normalize
        output = qkv / (qk_sum + 1e-6)
        
        # Reshape output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.proj(output)
        output = self.dropout(output)
        
        # Compute importance scores if needed
        importance = None
        if return_importance:
            importance = qk.sum(dim=1).mean(dim=1)  # Average across heads
            importance = F.softmax(importance, dim=-1)
        
        return output, importance


class TokenSelector(nn.Module):
    """Selects important visual tokens dynamically."""
    
    def __init__(self, config: FastVLMConfig):
        super().__init__()
        self.config = config
        self.importance_head = nn.Linear(config.vision_hidden_size, 1)
        
    def forward(
        self, 
        vision_features: torch.Tensor, 
        importance_scores: Optional[torch.Tensor] = None,
        num_tokens: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.LongTensor, torch.Tensor]:
        """
        Select important tokens from vision features.
        
        Args:
            vision_features: Vision features [batch_size, seq_len, hidden_size]
            importance_scores: Pre-computed importance scores
            num_tokens: Number of tokens to select
            
        Returns:
            selected_features: Selected vision features
            selected_indices: Indices of selected tokens
            scores: Token importance scores
        """
        batch_size, seq_len, hidden_size = vision_features.shape
        
        if num_tokens is None:
            num_tokens = min(self.config.num_vision_tokens, seq_len)
        
        # Compute importance scores if not provided
        if importance_scores is None:
            importance_scores = self.importance_head(vision_features).squeeze(-1)
            importance_scores = F.softmax(importance_scores, dim=-1)
        
        # Select top-k tokens
        scores, indices = torch.topk(importance_scores, num_tokens, dim=-1)
        
        # Gather selected features
        selected_features = torch.gather(
            vision_features,
            1,
            indices.unsqueeze(-1).expand(-1, -1, hidden_size)
        )
        
        return selected_features, indices, scores


class FastVLMModel(PreTrainedModel):
    """Fast Vision-Language Model with efficient attention and token selection."""
    
    config_class = FastVLMConfig
    
    def __init__(self, config: FastVLMConfig):
        super().__init__(config)
        self.config = config
        
        # Vision encoder
        self.vision_encoder = EfficientVisionEncoder(config)
        
        # Token selector
        self.token_selector = TokenSelector(config)
        
        # Vision-language projection
        self.vision_projection = nn.Sequential(
            nn.Linear(config.vision_hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.vision_dropout),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
        # Load language model
        self.language_model = AutoModelForCausalLM.from_pretrained(
            config.language_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Adapter layers for vision-language fusion
        self.cross_modal_layers = nn.ModuleList([
            CrossModalAdapter(config) for _ in range(4)
        ])
        
        # Initialize weights
        self.post_init()
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, FastVLMOutput]:
        """
        Forward pass of FastVLM model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            pixel_values: Input images
            labels: Labels for language modeling
            past_key_values: Cached key-value pairs
            use_cache: Whether to use caching
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a ModelOutput object
            
        Returns:
            Model outputs
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Process vision inputs if provided
        vision_features = None
        selected_indices = None
        token_importance_scores = None
        
        if pixel_values is not None:
            # Encode vision features
            vision_features, importance_scores = self.vision_encoder(pixel_values)
            
            # Select important tokens
            vision_features, selected_indices, token_importance_scores = self.token_selector(
                vision_features, importance_scores
            )
            
            # Project to language model dimension
            vision_features = self.vision_projection(vision_features)
            
            # Prepare vision tokens for language model
            batch_size = vision_features.shape[0]
            vision_embeds = vision_features
            
            # Get language model embeddings
            if input_ids is not None:
                inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
                
                # Concatenate vision and language embeddings
                inputs_embeds = torch.cat([vision_embeds, inputs_embeds], dim=1)
                
                # Update attention mask
                vision_attention = torch.ones(
                    batch_size, vision_embeds.shape[1], 
                    device=vision_embeds.device, 
                    dtype=attention_mask.dtype if attention_mask is not None else torch.long
                )
                
                if attention_mask is not None:
                    attention_mask = torch.cat([vision_attention, attention_mask], dim=1)
                else:
                    attention_mask = vision_attention
            else:
                inputs_embeds = vision_embeds
        else:
            inputs_embeds = None
        
        # Forward through language model with cross-modal adapters
        outputs = self.language_model(
            input_ids=input_ids if inputs_embeds is None else None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Apply cross-modal adapters
        if vision_features is not None and hasattr(outputs, 'hidden_states'):
            hidden_states = list(outputs.hidden_states)
            
            # Apply adapters to intermediate layers
            adapter_positions = [6, 12, 18, 24]  # Adjust based on model depth
            for i, pos in enumerate(adapter_positions):
                if pos < len(hidden_states):
                    hidden_states[pos] = self.cross_modal_layers[i](
                        hidden_states[pos], vision_features
                    )
        
        if not return_dict:
            output = (outputs.logits,) + outputs[1:]
            return output
        
        return FastVLMOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
            vision_features=vision_features,
            selected_tokens=selected_indices,
            token_importance_scores=token_importance_scores,
        )
    
    def generate(self, *args, **kwargs):
        """Generate text using the language model."""
        return self.language_model.generate(*args, **kwargs)


class CrossModalAdapter(nn.Module):
    """Adapter layer for cross-modal fusion."""
    
    def __init__(self, config: FastVLMConfig):
        super().__init__()
        self.down_proj = nn.Linear(config.hidden_size, config.hidden_size // 4)
        self.up_proj = nn.Linear(config.hidden_size // 4, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.mlp_dropout)
        
    def forward(self, hidden_states: torch.Tensor, vision_features: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        
        # Down project
        hidden_states = self.down_proj(hidden_states)
        hidden_states = F.gelu(hidden_states)
        
        # Add vision information
        vision_context = vision_features.mean(dim=1, keepdim=True)
        vision_context = self.down_proj(vision_context)
        hidden_states = hidden_states + vision_context.expand_as(hidden_states)
        
        # Up project
        hidden_states = self.up_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Residual connection
        hidden_states = self.norm(residual + hidden_states)
        
        return hidden_states
