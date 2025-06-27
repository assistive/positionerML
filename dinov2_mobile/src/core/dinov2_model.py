"""
DINOv2 Model Handler for Mobile Deployment
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path


class DINOv2Mobile:
    """Mobile-optimized DINOv2 model handler."""
    
    def __init__(self, model_name: str = "dinov2_vits14", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.input_size = (224, 224)
        self.patch_size = 14
        
    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load DINOv2 model."""
        if model_path:
            self.model = torch.load(model_path, map_location=self.device)
        else:
            # Load from torch.hub
            self.model = torch.hub.load('facebookresearch/dinov2', self.model_name)
        
        self.model.eval()
        self.model.to(self.device)
        
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for DINOv2 inference."""
        # Resize to 224x224
        from PIL import Image
        import torchvision.transforms as transforms
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        return transform(image).unsqueeze(0).to(self.device)
    
    def extract_features(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features using DINOv2."""
        with torch.no_grad():
            features = self.model(image)
            
            # Extract different feature representations
            cls_token = features[:, 0]  # Classification token
            patch_tokens = features[:, 1:]  # Patch tokens
            
            return {
                "cls_features": cls_token,
                "patch_features": patch_tokens,
                "all_features": features
            }
    
    def get_model_info(self) -> Dict:
        """Get model information for deployment."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_name": self.model_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "input_size": self.input_size,
            "patch_size": self.patch_size,
            "device": self.device
        }
