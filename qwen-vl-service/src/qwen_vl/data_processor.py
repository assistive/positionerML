"""
Qwen 2.5-VL Data Processor

Handles data preprocessing and loading for Qwen 2.5-VL models.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Tuple
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class QwenVLDataset(Dataset):
    """Dataset class for Qwen 2.5-VL training data."""
    
    def __init__(
        self,
        data_path: str,
        processor,
        max_length: int = 512,
        image_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to dataset file
            processor: Qwen VL processor
            max_length: Maximum sequence length
            image_size: Target image size
        """
        self.data_path = Path(data_path)
        self.processor = processor
        self.max_length = max_length
        self.image_size = image_size
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict]:
        """Load data from file."""
        if self.data_path.suffix == '.json':
            with open(self.data_path, 'r') as f:
                return json.load(f)
        elif self.data_path.suffix == '.jsonl':
            data = []
            with open(self.data_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            return data
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data item."""
        item = self.data[idx]
        
        # Process the conversation
        messages = item.get('messages', item.get('conversation', []))
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        
        # Process images/videos if present
        images = []
        videos = []
        
        for message in messages:
            if isinstance(message.get('content'), list):
                for content_item in message['content']:
                    if content_item.get('type') == 'image_url':
                        image = self._load_image(content_item['image_url']['url'])
                        if image:
                            images.append(image)
                    elif content_item.get('type') == 'video':
                        video = self._load_video(content_item['video'])
                        if video:
                            videos.append(video)
        
        # Tokenize
        inputs = self.processor(
            text=[text],
            images=images if images else None,
            videos=videos if videos else None,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Squeeze batch dimension
        return {k: v.squeeze(0) for k, v in inputs.items()}
    
    def _load_image(self, image_path_or_url: str) -> Optional[Image.Image]:
        """Load image from path or URL."""
        try:
            if image_path_or_url.startswith('data:image'):
                # Base64 encoded image
                image_data = image_path_or_url.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_bytes))
            elif image_path_or_url.startswith('http'):
                # URL - would need requests in real implementation
                logger.warning("URL image loading not implemented")
                return None
            else:
                # Local file path
                image = Image.open(image_path_or_url)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if needed
            if image.size != self.image_size:
                image = image.resize(self.image_size, Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path_or_url}: {e}")
            return None
    
    def _load_video(self, video_path: str) -> Optional[List[Image.Image]]:
        """Load video frames from path."""
        try:
            if isinstance(video_path, list):
                # List of image paths representing video frames
                frames = []
                for frame_path in video_path:
                    frame = self._load_image(frame_path)
                    if frame:
                        frames.append(frame)
                return frames if frames else None
            else:
                # Video file path
                cap = cv2.VideoCapture(video_path)
                frames = []
                
                # Sample frames (simplified - real implementation would be more sophisticated)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                sample_rate = max(1, total_frames // 32)  # Sample up to 32 frames
                
                frame_idx = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_idx % sample_rate == 0:
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(frame_rgb)
                        
                        # Resize
                        if image.size != self.image_size:
                            image = image.resize(self.image_size, Image.Resampling.LANCZOS)
                        
                        frames.append(image)
                    
                    frame_idx += 1
                
                cap.release()
                return frames if frames else None
                
        except Exception as e:
            logger.error(f"Error loading video {video_path}: {e}")
            return None

class QwenVLDataProcessor:
    """Data processing utilities for Qwen 2.5-VL."""
    
    def __init__(self, processor):
        """Initialize with Qwen VL processor."""
        self.processor = processor
    
    def create_dataloader(
        self,
        dataset: QwenVLDataset,
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 4
    ) -> DataLoader:
        """Create a DataLoader for the dataset."""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Custom collate function for batching."""
        # This is a simplified collate function
        # Real implementation would handle variable-length sequences properly
        
        collated = {}
        for key in batch[0].keys():
            tensors = [item[key] for item in batch]
            
            if key == 'input_ids' or key == 'attention_mask':
                # Pad sequences
                max_len = max(t.size(0) for t in tensors)
                padded_tensors = []
                
                for tensor in tensors:
                    pad_size = max_len - tensor.size(0)
                    if pad_size > 0:
                        if key == 'input_ids':
                            pad_value = self.processor.tokenizer.pad_token_id
                        else:
                            pad_value = 0
                        
                        padded = torch.cat([
                            tensor,
                            torch.full((pad_size,), pad_value, dtype=tensor.dtype)
                        ])
                    else:
                        padded = tensor
                    
                    padded_tensors.append(padded)
                
                collated[key] = torch.stack(padded_tensors)
            else:
                # For other tensors, try to stack directly
                try:
                    collated[key] = torch.stack(tensors)
                except:
                    # If stacking fails, keep as list
                    collated[key] = tensors
        
        return collated
    
    def prepare_inference_input(
        self,
        text: str,
        image_path: Optional[str] = None,
        video_path: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """Prepare input for inference."""
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        
        # Add image if provided
        if image_path:
            image_content = {
                "type": "image_url",
                "image_url": {"url": f"file://{image_path}"}
            }
            messages[0]["content"].append(image_content)
        
        # Add video if provided
        if video_path:
            video_content = {
                "type": "video",
                "video": video_path
            }
            messages[0]["content"].append(video_content)
        
        # Process with the processor
        text_formatted = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # This would need proper image/video loading in real implementation
        inputs = self.processor(
            text=[text_formatted],
            return_tensors="pt"
        )
        
        return inputs
