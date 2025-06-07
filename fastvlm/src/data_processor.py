# fastvlm/src/data_processor.py

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import logging
from transformers import AutoTokenizer, AutoImageProcessor
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from dataclasses import dataclass
from functools import lru_cache
import h5py
import lmdb
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data processing."""
    image_size: int = 224
    max_length: int = 512
    pad_token_id: int = 0
    use_augmentation: bool = True
    augmentation_prob: float = 0.8
    cache_dir: Optional[str] = None
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    use_lmdb: bool = False
    use_hdf5: bool = False


class FastVLMDataset(Dataset):
    """Efficient dataset for vision-language tasks."""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: AutoTokenizer,
        image_processor: AutoImageProcessor,
        config: DataConfig,
        split: str = "train",
        lazy_load: bool = True
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.config = config
        self.split = split
        self.lazy_load = lazy_load
        
        # Load data
        self.data = self._load_data()
        
        # Setup augmentation
        self.transform = self._get_transforms()
        
        # Setup cache
        self.cache = {}
        if config.cache_dir:
            self.cache_dir = Path(config.cache_dir) / split
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
        # Setup LMDB if enabled
        if config.use_lmdb:
            self._setup_lmdb()
            
        # Setup HDF5 if enabled
        if config.use_hdf5:
            self._setup_hdf5()
            
        logger.info(f"Loaded {len(self.data)} samples for {split} split")
    
    def _load_data(self) -> List[Dict]:
        """Load data from various formats."""
        data = []
        
        if self.data_path.is_file():
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
            # Load from directory
            for file_path in self.data_path.glob("*.json"):
                with open(file_path, 'r') as f:
                    data.extend(json.load(f))
        
        return data
    
    def _get_transforms(self) -> A.Compose:
        """Get image augmentation transforms."""
        if self.split == "train" and self.config.use_augmentation:
            return A.Compose([
                A.RandomResizedCrop(
                    height=self.config.image_size,
                    width=self.config.image_size,
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.333)
                ),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1,
                    p=0.8
                ),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], p=self.config.augmentation_prob)
        else:
            return A.Compose([
                A.Resize(
                    height=self.config.image_size,
                    width=self.config.image_size
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    
    def _setup_lmdb(self):
        """Setup LMDB database for fast loading."""
        self.lmdb_path = self.cache_dir / "images.lmdb"
        self.lmdb_env = lmdb.open(
            str(self.lmdb_path),
            map_size=1099511627776,  # 1TB
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
    
    def _setup_hdf5(self):
        """Setup HDF5 file for fast loading."""
        self.hdf5_path = self.cache_dir / "images.h5"
        if self.hdf5_path.exists():
            self.hdf5_file = h5py.File(self.hdf5_path, 'r')
    
    @lru_cache(maxsize=1024)
    def _load_image(self, image_path: str) -> Image.Image:
        """Load image with caching."""
        if self.config.use_lmdb and hasattr(self, 'lmdb_env'):
            # Load from LMDB
            with self.lmdb_env.begin(write=False) as txn:
                image_bytes = txn.get(image_path.encode())
                if image_bytes:
                    image = pickle.loads(image_bytes)
                    return Image.fromarray(image)
        
        if self.config.use_hdf5 and hasattr(self, 'hdf5_file'):
            # Load from HDF5
            if image_path in self.hdf5_file:
                image_array = self.hdf5_file[image_path][:]
                return Image.fromarray(image_array)
        
        # Load from disk
        image = Image.open(image_path).convert('RGB')
        return image
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Check cache
        if idx in self.cache:
            return self.cache[idx]
        
        item = self.data[idx]
        
        # Load and process image
        image = self._load_image(item['image_path'])
        image_array = np.array(image)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image_array)
            pixel_values = augmented['image']
        else:
            pixel_values = torch.from_numpy(image_array).permute(2, 0, 1)
        
        # Process text
        conversations = item.get('conversations', [])
        if conversations:
            # Format conversations
            text = self._format_conversations(conversations)
        else:
            text = item.get('caption', '')
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        result = {
            'pixel_values': pixel_values,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0).clone()
        }
        
        # Cache if enabled
        if self.config.cache_dir and len(self.cache) < 10000:
            self.cache[idx] = result
        
        return result
    
    def _format_conversations(self, conversations: List[Dict]) -> str:
        """Format conversations into a single string."""
        formatted = []
        for conv in conversations:
            role = conv.get('from', 'user')
            value = conv.get('value', '')
            
            if role == 'human' or role == 'user':
                formatted.append(f"Human: {value}")
            elif role == 'assistant' or role == 'gpt':
                formatted.append(f"Assistant: {value}")
        
        return "\n".join(formatted)


class FastVLMCollator:
    """Custom collator for batching with dynamic padding."""
    
    def __init__(self, tokenizer: AutoTokenizer, config: DataConfig):
        self.tokenizer = tokenizer
        self.config = config
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # Stack pixel values
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        
        # Find max length in batch for dynamic padding
        max_length = max(item['input_ids'].shape[0] for item in batch)
        max_length = min(max_length, self.config.max_length)
        
        # Pad sequences
        input_ids = []
        attention_mask = []
        labels = []
        
        for item in batch:
            seq_len = item['input_ids'].shape[0]
            
            if seq_len < max_length:
                # Pad
                padding_length = max_length - seq_len
                input_ids.append(torch.cat([
                    item['input_ids'],
                    torch.full((padding_length,), self.config.pad_token_id)
                ]))
                attention_mask.append(torch.cat([
                    item['attention_mask'],
                    torch.zeros(padding_length, dtype=torch.long)
                ]))
                labels.append(torch.cat([
                    item['labels'],
                    torch.full((padding_length,), -100)  # Ignore padding in loss
                ]))
            else:
                # Truncate
                input_ids.append(item['input_ids'][:max_length])
                attention_mask.append(item['attention_mask'][:max_length])
                labels.append(item['labels'][:max_length])
        
        return {
            'pixel_values': pixel_values,
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_mask),
            'labels': torch.stack(labels)
        }


class DataProcessor:
    """Main data processor for FastVLM."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        
    def create_datasets(
        self,
        train_path: Optional[str] = None,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        image_processor: Optional[AutoImageProcessor] = None
    ) -> Dict[str, FastVLMDataset]:
        """Create datasets for training, validation, and testing."""
        datasets = {}
        
        if train_path:
            datasets['train'] = FastVLMDataset(
                train_path,
                tokenizer,
                image_processor,
                self.config,
                split='train'
            )
        
        if val_path:
            datasets['val'] = FastVLMDataset(
                val_path,
                tokenizer,
                image_processor,
                self.config,
                split='val'
            )
        
        if test_path:
            datasets['test'] = FastVLMDataset(
                test_path,
                tokenizer,
                image_processor,
                self.config,
                split='test'
            )
        
        return datasets
    
    def create_dataloaders(
        self,
        datasets: Dict[str, FastVLMDataset],
        batch_size: int,
        distributed: bool = False,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None
    ) -> Dict[str, DataLoader]:
        """Create data loaders from datasets."""
        dataloaders = {}
        collator = FastVLMCollator(
            datasets[list(datasets.keys())[0]].tokenizer,
            self.config
        )
        
        for split, dataset in datasets.items():
            if distributed:
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=num_replicas,
                    rank=rank,
                    shuffle=(split == 'train')
                )
            else:
                sampler = None
            
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == 'train' and not distributed),
                sampler=sampler,
                num_workers=self.config.num_workers,
                collate_fn=collator,
                pin_memory=True,
                prefetch_factor=self.config.prefetch_factor,
                persistent_workers=self.config.persistent_workers
            )
        
        return dataloaders
    
    def preprocess_and_cache(
        self,
        data_path: str,
        output_dir: str,
        tokenizer: AutoTokenizer,
        image_processor: AutoImageProcessor,
        use_lmdb: bool = True,
        use_hdf5: bool = True
    ):
        """Preprocess and cache data for faster loading."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load data
        with open(data_path, 'r') as f:
            if data_path.endswith('.jsonl'):
                data = [json.loads(line) for line in f]
            else:
                data = json.load(f)
        
        # Process images
        if use_lmdb:
            self._create_lmdb_cache(data, output_path)
        
        if use_hdf5:
            self._create_hdf5_cache(data, output_path)
        
        # Process text
        self._preprocess_text(data, tokenizer, output_path)
        
        logger.info(f"Preprocessing complete. Cached data saved to {output_path}")
    
    def _create_lmdb_cache(self, data: List[Dict], output_path: Path):
        """Create LMDB cache for images."""
        lmdb_path = output_path / "images.lmdb"
        
        env = lmdb.open(
            str(lmdb_path),
            map_size=1099511627776  # 1TB
        )
        
        with env.begin(write=True) as txn:
            for item in data:
                image_path = item['image_path']
                image = Image.open(image_path).convert('RGB')
                image_array = np.array(image)
                
                # Serialize and store
                txn.put(
                    image_path.encode(),
                    pickle.dumps(image_array)
                )
        
        env.close()
        logger.info(f"Created LMDB cache at {lmdb_path}")
    
    def _create_hdf5_cache(self, data: List[Dict], output_path: Path):
        """Create HDF5 cache for images."""
        hdf5_path = output_path / "images.h5"
        
        with h5py.File(hdf5_path, 'w') as f:
            for item in data:
                image_path = item['image_path']
                image = Image.open(image_path).convert('RGB')
                image_array = np.array(image)
                
                # Store in HDF5
                f.create_dataset(
                    image_path,
                    data=image_array,
                    compression='gzip',
                    compression_opts=4
                )
        
        logger.info(f"Created HDF5 cache at {hdf5_path}")
    
    def _preprocess_text(self, data: List[Dict], tokenizer: AutoTokenizer, output_path: Path):
        """Preprocess and cache tokenized text."""
        tokenized_data = []
        
        for item in data:
            # Process conversations
            if 'conversations' in item:
                text = self._format_conversations(item['conversations'])
            else:
                text = item.get('caption', '')
            
            # Tokenize
            encoding = tokenizer(
                text,
                max_length=self.config.max_length,
                padding='max_length',
                truncation=True
            )
            
            tokenized_item = {
                'image_path': item['image_path'],
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask']
            }
            
            tokenized_data.append(tokenized_item)
        
        # Save tokenized data
        with open(output_path / "tokenized_data.json", 'w') as f:
            json.dump(tokenized_data, f)
        
        logger.info(f"Saved tokenized data to {output_path / 'tokenized_data.json'}")
    
    def _format_conversations(self, conversations: List[Dict]) -> str:
        """Format conversations into a single string."""
        formatted = []
        for conv in conversations:
            role = conv.get('from', 'user')
            value = conv.get('value', '')
            
            if role == 'human' or role == 'user':
                formatted.append(f"Human: {value}")
            elif role == 'assistant' or role == 'gpt':
                formatted.append(f"Assistant: {value}")
        
        return "\n".join(formatted)
