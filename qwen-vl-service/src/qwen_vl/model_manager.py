"""
Qwen 2.5-VL Model Manager

Handles model loading, optimization, and inference management.
"""

import torch
import yaml
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer
)
from qwen_vl_utils import process_vision_info
import logging

logger = logging.getLogger(__name__)

class QwenVLModelManager:
    """Manages Qwen 2.5-VL model loading and inference."""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """Initialize model manager with configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.device = self._get_device()
        
    def _load_config(self) -> Dict:
        """Load model configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_device(self) -> str:
        """Determine the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def load_model(self, variant: str = "qwen-2.5-vl-7b", **kwargs) -> None:
        """
        Load a specific Qwen 2.5-VL model variant.
        
        Args:
            variant: Model variant to load (3b, 7b, 32b, 72b)
            **kwargs: Additional arguments for model loading
        """
        try:
            model_config = self.config['model']['variants'][variant]
            model_id = model_config['model_id']
            
            logger.info(f"Loading Qwen 2.5-VL model: {model_id}")
            
            # Model loading arguments
            load_args = {
                "torch_dtype": kwargs.get("torch_dtype", "auto"),
                "device_map": kwargs.get("device_map", "auto"),
                "trust_remote_code": True
            }
            
            # Enable flash attention if specified
            if self.config['model']['text']['flash_attention']:
                load_args["attn_implementation"] = "flash_attention_2"
            
            # Load model
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, **load_args
            )
            
            # Load processor and tokenizer
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # Apply optimizations
            if self.config['model']['mobile']['compilation']['torch_compile']:
                self.model = torch.compile(self.model)
            
            logger.info(f"Model {variant} loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model {variant}: {e}")
            raise
    
    def process_input(
        self, 
        messages: List[Dict],
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None
    ) -> Dict:
        """
        Process input messages for the model.
        
        Args:
            messages: List of message dictionaries
            min_pixels: Minimum pixels for image processing
            max_pixels: Maximum pixels for image processing
            
        Returns:
            Processed input tensors
        """
        if not self.processor:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Set pixel limits from config if not provided
        vision_config = self.config['model']['vision']
        min_pixels = min_pixels or vision_config['min_pixels']
        max_pixels = max_pixels or vision_config['max_pixels']
        
        # Process vision information
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = inputs.to(self.device)
        
        return inputs
    
    def generate(
        self,
        messages: List[Dict],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """
        Generate response from the model.
        
        Args:
            messages: Input messages
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Process inputs
            inputs = self.process_input(messages)
            
            # Generation parameters
            generation_args = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": temperature > 0,
                "pad_token_id": self.tokenizer.eos_token_id,
                **kwargs
            }
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    **generation_args
                )
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in 
                zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            return output_text
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if not self.model:
            return {"status": "No model loaded"}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        
        return {
            "model_type": self.model.config.model_type,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": self.device,
            "precision": str(self.model.dtype),
            "memory_usage": torch.cuda.memory_allocated() if torch.cuda.is_available() else "N/A"
        }
