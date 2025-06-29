"""
MobileCLIP Model Downloader
Downloads pretrained models from Apple and Hugging Face
"""
import os
import requests
import hashlib
from pathlib import Path
from typing import Dict, Optional
import logging
from tqdm import tqdm
import yaml

class MobileCLIPDownloader:
    """Download MobileCLIP models from various sources."""
    
    def __init__(self, cache_dir: str = "./models/pretrained"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Official Apple model URLs
        self.model_urls = {
            "mobileclip_s0": "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s0.pt",
            "mobileclip_s1": "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s1.pt", 
            "mobileclip_s2": "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s2.pt",
            "mobileclip_b": "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_b.pt",
            "mobileclip_blt": "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt"
        }
        
        # Hugging Face model names
        self.hf_models = {
            "mobileclip_s0": "apple/MobileCLIP-S0",
            "mobileclip_s1": "apple/MobileCLIP-S1",
            "mobileclip_s2": "apple/MobileCLIP-S2", 
            "mobileclip_b": "apple/MobileCLIP-B"
        }
    
    def download_file(self, url: str, filepath: Path, chunk_size: int = 8192) -> bool:
        """Download file with progress bar."""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f, tqdm(
                desc=filepath.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logging.info(f"Downloaded {filepath}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to download {url}: {e}")
            if filepath.exists():
                filepath.unlink()
            return False
    
    def download_from_apple(self, model_name: str, force_download: bool = False) -> Optional[Path]:
        """Download model from Apple's official repository."""
        if model_name not in self.model_urls:
            logging.error(f"Model {model_name} not available from Apple")
            return None
        
        filepath = self.cache_dir / f"{model_name}.pt"
        
        if filepath.exists() and not force_download:
            logging.info(f"Model {model_name} already exists at {filepath}")
            return filepath
        
        url = self.model_urls[model_name]
        logging.info(f"Downloading {model_name} from Apple...")
        
        if self.download_file(url, filepath):
            return filepath
        return None
    
    def download_from_huggingface(self, model_name: str, force_download: bool = False) -> Optional[Path]:
        """Download model from Hugging Face."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            logging.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
            return None
        
        if model_name not in self.hf_models:
            logging.error(f"Model {model_name} not available on Hugging Face")
            return None
        
        try:
            hf_model_name = self.hf_models[model_name]
            filepath = hf_hub_download(
                repo_id=hf_model_name,
                filename="pytorch_model.bin",
                cache_dir=str(self.cache_dir),
                force_download=force_download
            )
            logging.info(f"Downloaded {model_name} from Hugging Face to {filepath}")
            return Path(filepath)
            
        except Exception as e:
            logging.error(f"Failed to download from Hugging Face: {e}")
            return None
    
    def download_model(self, model_name: str, source: str = "apple", force_download: bool = False) -> Optional[Path]:
        """Download model from specified source."""
        if source == "apple":
            return self.download_from_apple(model_name, force_download)
        elif source == "huggingface":
            return self.download_from_huggingface(model_name, force_download)
        else:
            logging.error(f"Unknown source: {source}")
            return None
    
    def download_all_models(self, source: str = "apple", force_download: bool = False) -> Dict[str, Optional[Path]]:
        """Download all available models."""
        results = {}
        
        if source == "apple":
            models = self.model_urls.keys()
        elif source == "huggingface":
            models = self.hf_models.keys()
        else:
            logging.error(f"Unknown source: {source}")
            return results
        
        for model_name in models:
            results[model_name] = self.download_model(model_name, source, force_download)
        
        return results
    
    def verify_model(self, model_path: Path) -> bool:
        """Verify downloaded model integrity."""
        if not model_path.exists():
            return False
        
        try:
            import torch
            torch.load(model_path, map_location='cpu')
            return True
        except Exception as e:
            logging.error(f"Model verification failed: {e}")
            return False
    
    def list_available_models(self) -> Dict[str, Dict]:
        """List all available models with their information."""
        models_info = {}
        
        for model_name in self.model_urls.keys():
            models_info[model_name] = {
                "apple_url": self.model_urls.get(model_name),
                "huggingface": self.hf_models.get(model_name),
                "local_path": self.cache_dir / f"{model_name}.pt",
                "downloaded": (self.cache_dir / f"{model_name}.pt").exists()
            }
        
        return models_info
