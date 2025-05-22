# internvl/src/model_downloader.py

import os
import torch
import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer, AutoModel, AutoConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelDownloader:
    """Downloads and manages InternVL models from Hugging Face."""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Initialize the model downloader.
        
        Args:
            config_path: Path to model configuration file
        """
        self.config = self.load_config(config_path)
        self.cache_dir = Path(self.config['model']['cache_dir'])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def download_model(self, 
                      model_name: Optional[str] = None,
                      force_download: bool = False) -> Tuple[str, str]:
        """
        Download InternVL model and tokenizer.
        
        Args:
            model_name: Hugging Face model identifier
            force_download: Whether to force re-download
            
        Returns:
            Tuple of (model_path, tokenizer_path)
        """
        if model_name is None:
            model_name = self.config['model']['variant']
            
        logger.info(f"Downloading InternVL model: {model_name}")
        
        # Create model-specific directory
        model_dir = self.cache_dir / model_name.replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download model files
            model_path = snapshot_download(
                repo_id=model_name,
                cache_dir=str(model_dir),
                force_download=force_download,
                resume_download=True
            )
            
            logger.info(f"Model downloaded to: {model_path}")
            return model_path, model_path
            
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            raise
    
    def load_model_and_tokenizer(self, 
                                model_path: str) -> Tuple[AutoModel, AutoTokenizer]:
        """
        Load model and tokenizer from downloaded path.
        
        Args:
            model_path: Path to downloaded model
            
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model from: {model_path}")
        
        try:
            # Load configuration
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True,
                use_fast=False
            )
            
            # Load model
            model = AutoModel.from_pretrained(
                model_path,
                config=config,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )
            
            logger.info("Model and tokenizer loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def get_model_info(self, model_path: str) -> Dict:
        """
        Get information about the downloaded model.
        
        Args:
            model_path: Path to model
            
        Returns:
            Dictionary with model information
        """
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        info = {
            "model_type": getattr(config, "model_type", "unknown"),
            "vocab_size": getattr(config, "vocab_size", 0),
            "hidden_size": getattr(config, "hidden_size", 0),
            "num_hidden_layers": getattr(config, "num_hidden_layers", 0),
            "num_attention_heads": getattr(config, "num_attention_heads", 0),
            "max_position_embeddings": getattr(config, "max_position_embeddings", 0),
        }
        
        # Calculate approximate model size
        try:
            model = AutoModel.from_pretrained(
                model_path, 
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            info.update({
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
            })
            
            del model  # Free memory
            
        except Exception as e:
            logger.warning(f"Could not calculate model size: {e}")
            
        return info
    
    def list_available_models(self) -> list:
        """List available InternVL models."""
        models = [
            "OpenGVLab/InternVL2-1B",
            "OpenGVLab/InternVL2-2B", 
            "OpenGVLab/InternVL2-4B",
            "OpenGVLab/InternVL2-8B",
            "OpenGVLab/InternVL2-26B",
            "OpenGVLab/InternVL2-40B",
            "OpenGVLab/InternVL2-Llama3-76B"
        ]
        return models
    
    def check_model_compatibility(self, model_path: str) -> Dict[str, bool]:
        """
        Check model compatibility for different deployment targets.
        
        Args:
            model_path: Path to model
            
        Returns:
            Dictionary with compatibility information
        """
        info = self.get_model_info(model_path)
        
        compatibility = {
            "mobile_ready": info.get("model_size_mb", float('inf')) < 2000,  # < 2GB
            "ios_compatible": True,  # Most models can be converted to CoreML
            "android_compatible": True,  # Most models can be converted to TFLite
            "quantization_recommended": info.get("model_size_mb", 0) > 500,  # > 500MB
        }
        
        return compatibility

