# src/model_downloader.py

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import yaml
import torch

from huggingface_hub import hf_hub_download, snapshot_download, login
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    BertTokenizer, BertModel, BertConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BERTModelDownloader:
    """Download and manage BERT models from Hugging Face Hub."""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Initialize model downloader.
        
        Args:
            config_path: Path to model configuration file
        """
        self.config = self.load_config(config_path)
        self.cache_dir = Path(self.config['model']['cache_dir'])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Available BERT models
        self.available_models = {
            'bert-base-uncased': {
                'repo_id': 'bert-base-uncased',
                'description': 'BERT base model (uncased)',
                'size_mb': 440,
                'hidden_size': 768,
                'num_layers': 12,
                'num_heads': 12
            },
            'bert-base-cased': {
                'repo_id': 'bert-base-cased',
                'description': 'BERT base model (cased)',
                'size_mb': 440,
                'hidden_size': 768,
                'num_layers': 12,
                'num_heads': 12
            },
            'bert-large-uncased': {
                'repo_id': 'bert-large-uncased',
                'description': 'BERT large model (uncased)',
                'size_mb': 1340,
                'hidden_size': 1024,
                'num_layers': 24,
                'num_heads': 16
            },
            'distilbert-base-uncased': {
                'repo_id': 'distilbert-base-uncased',
                'description': 'DistilBERT base model (lighter)',
                'size_mb': 250,
                'hidden_size': 768,
                'num_layers': 6,
                'num_heads': 12
            },
            'albert-base-v2': {
                'repo_id': 'albert-base-v2',
                'description': 'ALBERT base model (parameter efficient)',
                'size_mb': 45,
                'hidden_size': 768,
                'num_layers': 12,
                'num_heads': 12
            },
            'roberta-base': {
                'repo_id': 'roberta-base',
                'description': 'RoBERTa base model',
                'size_mb': 500,
                'hidden_size': 768,
                'num_layers': 12,
                'num_heads': 12
            }
        }
    
    def load_config(self, config_path: str) -> Dict:
        """Load model configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def list_available_models(self) -> List[str]:
        """List available BERT models."""
        return list(self.available_models.keys())
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get information about a specific model."""
        if model_name in self.available_models:
            return self.available_models[model_name]
        else:
            # Try to get info from Hugging Face
            try:
                config = AutoConfig.from_pretrained(model_name)
                return {
                    'repo_id': model_name,
                    'description': f'Custom model: {model_name}',
                    'hidden_size': getattr(config, 'hidden_size', 'Unknown'),
                    'num_layers': getattr(config, 'num_hidden_layers', 'Unknown'),
                    'num_heads': getattr(config, 'num_attention_heads', 'Unknown'),
                    'vocab_size': getattr(config, 'vocab_size', 'Unknown')
                }
            except Exception as e:
                logger.warning(f"Could not get info for {model_name}: {e}")
                return {}
    
    def download_model(self, 
                      model_name: str,
                      force_download: bool = False,
                      use_auth_token: bool = False) -> Tuple[str, str]:
        """
        Download BERT model and tokenizer.
        
        Args:
            model_name: Model name or Hugging Face model ID
            force_download: Force re-download even if cached
            use_auth_token: Use Hugging Face authentication token
            
        Returns:
            Tuple of (model_path, tokenizer_path)
        """
        logger.info(f"Downloading BERT model: {model_name}")
        
        # Get model repository ID
        if model_name in self.available_models:
            repo_id = self.available_models[model_name]['repo_id']
        else:
            repo_id = model_name
        
        # Create model-specific directory
        model_dir = self.cache_dir / model_name.replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if model already exists
        if not force_download and (model_dir / "config.json").exists():
            logger.info(f"Model already exists at {model_dir}")
            return str(model_dir), str(model_dir)
        
        try:
            # Download model files
            logger.info(f"Downloading from {repo_id}...")
            
            auth_token = None
            if use_auth_token:
                auth_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
                if not auth_token:
                    logger.info("No auth token found, attempting login...")
                    login()
            
            # Download using snapshot_download for complete model
            downloaded_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=str(model_dir.parent),
                local_dir=str(model_dir),
                token=auth_token,
                force_download=force_download
            )
            
            logger.info(f"Model downloaded to: {model_dir}")
            
            # Verify download
            self.verify_download(model_dir)
            
            # Save download metadata
            self.save_download_metadata(model_dir, model_name, repo_id)
            
            return str(model_dir), str(model_dir)
            
        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {e}")
            raise
    
    def verify_download(self, model_dir: Path):
        """Verify that the downloaded model is complete."""
        logger.info("Verifying model download...")
        
        required_files = ["config.json"]
        model_files = ["pytorch_model.bin", "model.safetensors"]
        tokenizer_files = ["tokenizer.json", "vocab.txt"]
        
        # Check required files
        for file in required_files:
            if not (model_dir / file).exists():
                raise FileNotFoundError(f"Required file {file} not found")
        
        # Check for at least one model file
        if not any((model_dir / file).exists() for file in model_files):
            raise FileNotFoundError("No model weights file found")
        
        # Check for tokenizer files
        tokenizer_exists = any((model_dir / file).exists() for file in tokenizer_files)
        if not tokenizer_exists:
            logger.warning("No tokenizer files found, may need separate tokenizer")
        
        logger.info("Model verification successful")
    
    def save_download_metadata(self, model_dir: Path, model_name: str, repo_id: str):
        """Save metadata about the downloaded model."""
        metadata = {
            'model_name': model_name,
            'repo_id': repo_id,
            'download_date': str(pd.Timestamp.now()),
            'model_info': self.get_model_info(model_name),
            'config_used': self.config
        }
        
        # Add model size information
        total_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
        metadata['total_size_mb'] = total_size / (1024 * 1024)
        
        # Save metadata
        metadata_path = model_dir / "download_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Download metadata saved to {metadata_path}")
    
    def load_model_and_tokenizer(self, 
                                model_path: str,
                                custom_vocab_path: Optional[str] = None) -> Tuple:
        """
        Load model and tokenizer from downloaded path.
        
        Args:
            model_path: Path to downloaded model
            custom_vocab_path: Path to custom vocabulary file
            
        Returns:
            Tuple of (model, tokenizer, config)
        """
        logger.info(f"Loading model from: {model_path}")
        
        try:
            # Load configuration
            config = AutoConfig.from_pretrained(model_path)
            
            # Load tokenizer
            if custom_vocab_path:
                logger.info(f"Using custom vocabulary: {custom_vocab_path}")
                tokenizer = BertTokenizer.from_pretrained(
                    model_path,
                    vocab_file=custom_vocab_path,
                    do_lower_case=self.config['vocabulary']['tokenization']['do_lower_case']
                )
                # Update config vocab size if different
                custom_vocab_size = len(tokenizer.vocab)
                if config.vocab_size != custom_vocab_size:
                    logger.info(f"Updating vocab size from {config.vocab_size} to {custom_vocab_size}")
                    config.vocab_size = custom_vocab_size
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Load model
            model = AutoModel.from_pretrained(
                model_path,
                config=config
            )
            
            # Resize token embeddings if vocab size changed
            if custom_vocab_path:
                model.resize_token_embeddings(len(tokenizer.vocab))
            
            logger.info("Model and tokenizer loaded successfully")
            
            # Print model info
            num_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            logger.info(f"Model parameters: {num_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            logger.info(f"Vocabulary size: {len(tokenizer.vocab):,}")
            
            return model, tokenizer, config
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def create_mobile_config(self, 
                           base_config: Dict,
                           model_path: str) -> Dict:
        """
        Create mobile-optimized model configuration.
        
        Args:
            base_config: Base model configuration
            model_path: Path to model
            
        Returns:
            Mobile-optimized configuration
        """
        logger.info("Creating mobile-optimized configuration...")
        
        mobile_config = base_config.copy()
        mobile_settings = self.config['model']['mobile']
        
        # Apply mobile optimizations
        if mobile_settings.get('hidden_size'):
            mobile_config['hidden_size'] = mobile_settings['hidden_size']
        
        if mobile_settings.get('num_hidden_layers'):
            mobile_config['num_hidden_layers'] = mobile_settings['num_hidden_layers']
        
        if mobile_settings.get('num_attention_heads'):
            mobile_config['num_attention_heads'] = mobile_settings['num_attention_heads']
        
        if mobile_settings.get('intermediate_size'):
            mobile_config['intermediate_size'] = mobile_settings['intermediate_size']
        
        # Save mobile config
        mobile_config_path = Path(model_path) / "mobile_config.json"
        with open(mobile_config_path, 'w') as f:
            json.dump(mobile_config, f, indent=2)
        
        logger.info(f"Mobile config saved to {mobile_config_path}")
        return mobile_config
    
    def check_model_compatibility(self, model_path: str) -> Dict[str, bool]:
        """
        Check model compatibility for mobile deployment.
        
        Args:
            model_path: Path to model
            
        Returns:
            Dictionary with compatibility information
        """
        try:
            config = AutoConfig.from_pretrained(model_path)
            
            # Get model size
            model_dir = Path(model_path)
            total_size_mb = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file()) / (1024 * 1024)
            
            # Check compatibility
            compatibility = {
                "mobile_ready": total_size_mb < self.config['mobile']['max_model_size_mb'],
                "ios_compatible": True,  # Most BERT models can be converted to CoreML
                "android_compatible": True,  # Most BERT models can be converted to TFLite
                "quantization_recommended": total_size_mb > 100,  # Recommend quantization for models > 100MB
                "size_acceptable": total_size_mb < 200,  # Flag if model is too large
                "vocab_size_reasonable": config.vocab_size < 50000  # Large vocabs may be problematic
            }
            
            return compatibility
            
        except Exception as e:
            logger.error(f"Error checking compatibility: {e}")
            return {
                "mobile_ready": False,
                "ios_compatible": False,
                "android_compatible": False,
                "quantization_recommended": True,
                "size_acceptable": False,
                "vocab_size_reasonable": False,
                "error": str(e)
            }
    
    def cleanup_cache(self, keep_recent: int = 3):
        """
        Clean up model cache, keeping only recent downloads.
        
        Args:
            keep_recent: Number of recent models to keep
        """
        logger.info("Cleaning up model cache...")
        
        model_dirs = [d for d in self.cache_dir.iterdir() if d.is_dir()]
        
        if len(model_dirs) <= keep_recent:
            logger.info("No cleanup needed")
            return
        
        # Sort by modification time
        model_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Remove older models
        for model_dir in model_dirs[keep_recent:]:
            logger.info(f"Removing old model: {model_dir.name}")
            shutil.rmtree(model_dir)
        
        logger.info(f"Cleanup complete. Kept {keep_recent} recent models")
    
    def get_recommended_model(self, use_case: str = "mobile") -> str:
        """
        Get recommended model based on use case.
        
        Args:
            use_case: Use case (mobile, accuracy, speed)
            
        Returns:
            Recommended model name
        """
        recommendations = {
            "mobile": "distilbert-base-uncased",  # Smallest and fastest
            "accuracy": "bert-base-uncased",      # Good balance
            "speed": "albert-base-v2",            # Parameter efficient
            "multilingual": "bert-base-multilingual-cased"
        }
        
        return recommendations.get(use_case, "bert-base-uncased")
    
    def compare_models(self, model_names: List[str]) -> Dict:
        """
        Compare multiple models.
        
        Args:
            model_names: List of model names to compare
            
        Returns:
            Comparison data
        """
        comparison = {
            'models': {},
            'summary': {}
        }
        
        for model_name in model_names:
            info = self.get_model_info(model_name)
            comparison['models'][model_name] = info
        
        # Generate summary
        models_data = list(comparison['models'].values())
        if models_data:
            comparison['summary'] = {
                'smallest_model': min(model_names, key=lambda x: comparison['models'][x].get('size_mb', float('inf'))),
                'largest_model': max(model_names, key=lambda x: comparison['models'][x].get('size_mb', 0)),
                'avg_size_mb': sum(m.get('size_mb', 0) for m in models_data) / len(models_data)
            }
        
        return comparison
