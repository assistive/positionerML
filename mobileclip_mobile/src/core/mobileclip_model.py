"""
MobileCLIP Model Wrapper
Unified interface for MobileCLIP models with mobile optimization
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from pathlib import Path
import logging

try:
    import mobileclip
    MOBILECLIP_AVAILABLE = True
except ImportError:
    MOBILECLIP_AVAILABLE = False
    logging.warning("MobileCLIP package not found. Install with: pip install git+https://github.com/apple/ml-mobileclip")

class MobileCLIPModel:
    """Wrapper for MobileCLIP models with mobile deployment features."""
    
    def __init__(self, model_name: str = "mobileclip_s0", pretrained_path: Optional[str] = None):
        self.model_name = model_name
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if MOBILECLIP_AVAILABLE:
            self._load_model(pretrained_path)
    
    def _load_model(self, pretrained_path: Optional[str] = None):
        """Load MobileCLIP model and preprocessing."""
        try:
            self.model, _, self.preprocess = mobileclip.create_model_and_transforms(
                self.model_name, 
                pretrained=pretrained_path
            )
            self.tokenizer = mobileclip.get_tokenizer(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logging.info(f"Successfully loaded {self.model_name}")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to feature vector."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        with torch.no_grad():
            image_features = self.model.encode_image(image.to(self.device))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    def encode_text(self, text: List[str]) -> torch.Tensor:
        """Encode text to feature vector."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        text_tokens = self.tokenizer(text)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens.to(self.device))
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def compute_similarity(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """Compute similarity between image and text features."""
        return (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    def zero_shot_classify(self, image: torch.Tensor, text_labels: List[str]) -> Dict[str, float]:
        """Perform zero-shot classification."""
        image_features = self.encode_image(image)
        text_features = self.encode_text(text_labels)
        similarities = self.compute_similarity(image_features, text_features)
        
        results = {}
        for i, label in enumerate(text_labels):
            results[label] = float(similarities[0, i])
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if self.model is None:
            return {}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_name": self.model_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "input_size": [224, 224, 3],
            "text_max_length": 77
        }
    
    def prepare_for_mobile(self) -> nn.Module:
        """Prepare model for mobile deployment."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Reparameterize model for inference
        try:
            from mobileclip.modules.common.mobileone import reparameterize_model
            mobile_model = reparameterize_model(self.model)
            mobile_model.eval()
            return mobile_model
        except ImportError:
            logging.warning("Reparameterization not available, returning original model")
            return self.model
