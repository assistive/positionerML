# internvl/src/__init__.py

"""
InternVL Training and Deployment Package

This package provides utilities for:
- Downloading and managing InternVL models
- Fine-tuning with LoRA/QLoRA
- Converting models for mobile deployment
- Deploying to iOS and Android platforms
"""

__version__ = "1.0.0"
__author__ = "Your Organization"

from .model_downloader import ModelDownloader
from .data_processor import DataProcessor
from .trainer import InternVLTrainer
from .fine_tuner import InternVLFineTuner
from .mobile_converter import MobileConverter
from .deployment_utils import DeploymentUtils

__all__ = [
    "ModelDownloader",
    "DataProcessor", 
    "InternVLTrainer",
    "InternVLFineTuner",
    "MobileConverter",
    "DeploymentUtils"
]
