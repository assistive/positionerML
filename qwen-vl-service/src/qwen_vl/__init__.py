"""
Qwen 2.5-VL Integration Package

This package provides a service wrapper for Qwen 2.5-VL models,
including mobile deployment capabilities and optimization features.
"""

from .model_manager import QwenVLModelManager
from .service import QwenVLService
from .mobile_converter import QwenVLMobileConverter
from .data_processor import QwenVLDataProcessor

__version__ = "1.0.0"
__all__ = [
    "QwenVLModelManager",
    "QwenVLService", 
    "QwenVLMobileConverter",
    "QwenVLDataProcessor"
]
