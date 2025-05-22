# internvl/src/data_processor.py

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InternVLDataset(Dataset):
    """Dataset class for InternVL vision-language tasks."""
    
    def __init__(self, 
                 data_path: str,
                 tokenizer: AutoTokenizer,
                 max_length: int = 2048,
                 image_size: int = 448,
                 mode: str = "train"):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to dataset directory or file
            tokenizer: Pretrained tokenizer
            max_length: Maximum sequence length
            image_size: Image size for processing
            mode: Dataset mode ('train', 'val', 'test')
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_size = image_size
        self.mode = mode
        
        self.data = self.load_data()
        logger.info(f"Loaded {len(self.data)} samples for {mode} mode")
    
    def load_data(self) -> List[Dict]:
        """Load data from various formats."""
        data = []
        
        if self.data_path.is_file():
            # Single file
            if self.data_path.suffix == '.json':
                with open(self.data_path, 'r') as f:
                    data = json.load(f)
            elif self.data_path.suffix == '.jsonl':
                with open(self.data_path, 'r') as f:
                    data = [json.loads(line) for line in f]
            elif self.data_path.suffix == '.csv':
                df = pd.read_csv(self.data_path)
                data = df.to_dict('records')
        else:
            # Directory with multiple files
            for file_path in self.data_path.glob("*.json"):
                with open(file_path, 'r') as f:
                    data.extend(json.load(f))
                    
            for file_path in self.data_path.glob("*.jsonl"):
                with open(file_path, 'r') as f:
                    data.extend([json.loads(line) for line in f])
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data sample."""
        item = self.data[idx]
        
        # Load and process image
        image = self.load_image(item.get('image_path', ''))
        
        # Process text
        question = item.get('question', '')
        answer = item.get('answer', '')
        
        # Create conversation format
        conversation = f"<image>\nUser: {question}\nAssistant: {answer}"
        
        # Tokenize
        inputs = self.tokenizer(
            conversation,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'image': image,
            'labels': inputs['input_ids'].squeeze(0).clone()
        }
    
    def load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image."""
        if not image_path or not os.path.exists(image_path):
            # Return dummy image if path is invalid
            return torch.zeros(3, self.image_size, self.image_size)
        
        try:
            image = Image.open(image_path).convert('RGB')
            image = image.resize((self.image_size, self.image_size))
            
            # Convert to tensor and normalize
            image_array = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
            
            # Apply normalization (ImageNet stats)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std
            
            return image_tensor
            
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            return torch.zeros(3, self.image_size, self.image_size)

class DataProcessor:
    """Processes data for InternVL training and fine-tuning."""
    
    def __init__(self, tokenizer: AutoTokenizer, config: Dict):
        """
        Initialize data processor.
        
        Args:
            tokenizer: Pretrained tokenizer
            config: Configuration dictionary
        """
        self.tokenizer = tokenizer
        self.config = config
        
    def create_dataloaders(self, 
                          train_path: str,
                          val_path: str = None,
                          test_path: str = None,
                          batch_size: int = 4,
                          num_workers: int = 4) -> Dict[str, DataLoader]:
        """
        Create data loaders for training, validation, and testing.
        
        Args:
            train_path: Path to training data
            val_path: Path to validation data
            test_path: Path to test data
            batch_size: Batch size
            num_workers: Number of worker processes
            
        Returns:
            Dictionary of data loaders
        """
        dataloaders = {}
        
        # Training data loader
        train_dataset = InternVLDataset(
            train_path, 
            self.tokenizer,
            max_length=self.config.get('max_length', 2048),
            image_size=self.config.get('image_size', 448),
            mode='train'
        )
        
        dataloaders['train'] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
        
        # Validation data loader
        if val_path:
            val_dataset = InternVLDataset(
                val_path,
                self.tokenizer,
                max_length=self.config.get('max_length', 2048),
                image_size=self.config.get('image_size', 448),
                mode='val'
            )
            
            dataloaders['val'] = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=self.collate_fn
            )
        
        # Test data loader
        if test_path:
            test_dataset = InternVLDataset(
                test_path,
                self.tokenizer,
                max_length=self.config.get('max_length', 2048),
                image_size=self.config.get('image_size', 448),
                mode='test'
            )
            
            dataloaders['test'] = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=self.collate_fn
            )
        
        return dataloaders
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Custom collate function for batching."""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        images = torch.stack([item['image'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'images': images,
            'labels': labels
        }
    
    def prepare_imu_context_data(self, 
                               imu_data_path: str,
                               image_data_path: str,
                               output_path: str):
        """
        Prepare data combining IMU context with visual data.
        
        Args:
            imu_data_path: Path to IMU data CSV
            image_data_path: Path to images directory
            output_path: Output path for processed data
        """
        logger.info("Preparing IMU context data...")
        
        # Load IMU data
        imu_df = pd.read_csv(imu_data_path)
        
        # Create training examples
        examples = []
        
        for idx, row in imu_df.iterrows():
            # Extract IMU features
            accel_x, accel_y, accel_z = row['accel_x'], row['accel_y'], row['accel_z']
            gyro_x, gyro_y, gyro_z = row['gyro_x'], row['gyro_y'], row['gyro_z']
            
            # Create context description
            context = self.create_driving_context(
                accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
            )
            
            # Find corresponding image
            timestamp = row.get('timestamp', idx)
            image_path = os.path.join(image_data_path, f"frame_{timestamp}.jpg")
            
            if os.path.exists(image_path):
                example = {
                    'image_path': image_path,
                    'question': f"Analyze this driving scene given the current vehicle motion: {context}",
                    'answer': "Based on the sensor data and visual scene, I can provide driving assistance and safety recommendations.",
                    'imu_data': {
                        'accel': [accel_x, accel_y, accel_z],
                        'gyro': [gyro_x, gyro_y, gyro_z]
                    }
                }
                examples.append(example)
        
        # Save processed data
        with open(output_path, 'w') as f:
            json.dump(examples, f, indent=2)
        
        logger.info(f"Saved {len(examples)} examples to {output_path}")
    
    def create_driving_context(self, 
                             accel_x: float, accel_y: float, accel_z: float,
                             gyro_x: float, gyro_y: float, gyro_z: float) -> str:
        """Create natural language description of driving context from IMU data."""
        
        context_parts = []
        
        # Analyze acceleration
        if abs(accel_x) > 2.0:
            if accel_x > 0:
                context_parts.append("accelerating forward")
            else:
                context_parts.append("braking")
        
        if abs(accel_y) > 1.5:
            if accel_y > 0:
                context_parts.append("turning left")
            else:
                context_parts.append("turning right")
        
        # Analyze rotation
        if abs(gyro_z) > 0.5:
            context_parts.append("changing direction")
        
        # Overall motion assessment
        total_accel = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        if total_accel > 12:  # Above gravity + some motion
            context_parts.append("dynamic driving conditions")
        else:
            context_parts.append("steady driving")
        
        if not context_parts:
            return "normal driving conditions"
        
        return ", ".join(context_parts)

