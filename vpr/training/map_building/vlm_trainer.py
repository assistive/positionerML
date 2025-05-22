# vpr/training/models/vlm/vlm_trainer.py

import os
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import wandb
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor

from ...utils.logging import setup_logger
from ...utils.checkpointing import CheckpointManager
from ...utils.distributed import setup_distributed, is_main_process
from ...data.preprocessing.augmentation import get_navigation_augmentations

logger = setup_logger(__name__)


@dataclass
class VLMTrainingConfig:
    """Configuration for VLM training"""
    # Model
    model_name: str = "microsoft/MobileVLM-1B"
    pretrained: bool = True
    freeze_backbone: bool = False
    
    # Navigation-specific heads
    add_navigation_head: bool = True
    add_condition_heads: bool = True
    navigation_features_dim: int = 512
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 30
    warmup_steps: int = 1000
    
    # Multi-resolution
    input_resolutions: List[Tuple[int, int]] = None
    multi_resolution_fusion: str = "attention"  # "attention", "concat", "average"
    
    # Optimization
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: float = 1.0
    
    # Regularization
    dropout: float = 0.1
    label_smoothing: float = 0.1
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    wandb_project: str = "vpr-vlm-training"
    
    def __post_init__(self):
        if self.input_resolutions is None:
            self.input_resolutions = [(320, 240), (640, 480)]


class NavigationVLMDataset(Dataset):
    """Dataset for training VLM on navigation tasks"""
    
    def __init__(self, data_path: str, config: VLMTrainingConfig, 
                 split: str = "train", transform=None):
        self.data_path = data_path
        self.config = config
        self.split = split
        self.transform = transform or get_navigation_augmentations(split == "train")
        
        # Load data
        self.samples = self._load_samples()
        logger.info(f"Loaded {len(self.samples)} samples for {split}")
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load dataset samples"""
        samples = []
        
        # Load from different sources based on dataset type
        metadata_path = os.path.join(self.data_path, f"{self.split}_metadata.json")
        
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            for item in metadata:
                sample = {
                    "image_path": os.path.join(self.data_path, item["image"]),
                    "caption": item.get("caption", ""),
                    "position": np.array(item.get("position", [0, 0, 0])),
                    "condition": item.get("condition", "day"),
                    "navigation_labels": item.get("navigation_labels", {}),
                    "semantic_labels": item.get("semantic_labels", [])
                }
                samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Load image
        from PIL import Image
        image = Image.open(sample["image_path"]).convert("RGB")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Create multi-resolution versions
        images = {}
        for res in self.config.input_resolutions:
            resized = image.resize(res, Image.LANCZOS)
            images[f"{res[0]}x{res[1]}"] = np.array(resized)
        
        return {
            "images": images,
            "caption": sample["caption"],
            "position": sample["position"],
            "condition": sample["condition"],
            "navigation_labels": sample["navigation_labels"],
            "semantic_labels": sample["semantic_labels"]
        }


class NavigationVLM(nn.Module):
    """VLM enhanced for navigation tasks"""
    
    def __init__(self, config: VLMTrainingConfig):
        super().__init__()
        self.config = config
        
        # Load base VLM
        self.base_model = AutoModel.from_pretrained(
            config.model_name,
            trust_remote_code=True
        )
        
        # Get hidden size from base model
        self.hidden_size = self.base_model.config.hidden_size
        
        # Freeze backbone if specified
        if config.freeze_backbone:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Navigation-specific head
        if config.add_navigation_head:
            self.navigation_head = nn.Sequential(
                nn.Linear(self.hidden_size, config.navigation_features_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.navigation_features_dim, config.navigation_features_dim)
            )
        
        # Condition-specific heads (day/night, weather, etc.)
        if config.add_condition_heads:
            self.condition_heads = nn.ModuleDict({
                "time": nn.Linear(self.hidden_size, 3),  # day, dusk/dawn, night
                "weather": nn.Linear(self.hidden_size, 5),  # clear, cloudy, rain, snow, fog
                "season": nn.Linear(self.hidden_size, 4)  # spring, summer, fall, winter
            })
        
        # Multi-resolution fusion
        if config.multi_resolution_fusion == "attention":
            self.resolution_attention = nn.MultiheadAttention(
                embed_dim=self.hidden_size,
                num_heads=8,
                dropout=config.dropout
            )
        elif config.multi_resolution_fusion == "concat":
            num_resolutions = len(config.input_resolutions)
            self.resolution_fusion = nn.Linear(
                self.hidden_size * num_resolutions,
                self.hidden_size
            )
    
    def forward(self, images: Dict[str, torch.Tensor], 
                captions: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model
        
        Args:
            images: Dictionary of resolution -> image tensors
            captions: Optional text captions
            
        Returns:
            Dictionary of outputs including navigation features
        """
        outputs = {}
        resolution_features = []
        
        # Process each resolution
        for resolution, image_batch in images.items():
            # Get features from base model
            if captions:
                base_outputs = self.base_model(
                    pixel_values=image_batch,
                    input_ids=captions
                )
            else:
                base_outputs = self.base_model(pixel_values=image_batch)
            
            # Extract pooled features
            if hasattr(base_outputs, "pooler_output"):
                features = base_outputs.pooler_output
            else:
                features = base_outputs.last_hidden_state.mean(dim=1)
            
            resolution_features.append(features)
        
        # Fuse multi-resolution features
        if self.config.multi_resolution_fusion == "attention":
            # Stack features for attention
            stacked_features = torch.stack(resolution_features, dim=1)
            fused_features, _ = self.resolution_attention(
                stacked_features, stacked_features, stacked_features
            )
            fused_features = fused_features.mean(dim=1)
        elif self.config.multi_resolution_fusion == "concat":
            fused_features = torch.cat(resolution_features, dim=-1)
            fused_features = self.resolution_fusion(fused_features)
        else:  # average
            fused_features = torch.stack(resolution_features, dim=0).mean(dim=0)
        
        outputs["features"] = fused_features
        
        # Navigation features
        if self.config.add_navigation_head:
            outputs["navigation_features"] = self.navigation_head(fused_features)
        
        # Condition predictions
        if self.config.add_condition_heads:
            outputs["condition_predictions"] = {
                name: head(fused_features)
                for name, head in self.condition_heads.items()
            }
        
        return outputs


class VLMTrainer:
    """Trainer for navigation-enhanced VLM"""
    
    def __init__(self, config: VLMTrainingConfig, 
                 model: Optional[NavigationVLM] = None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = model or NavigationVLM(config)
        self.model.to(self.device)
        
        # Initialize components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Losses
        self.navigation_loss = nn.MSELoss()
        self.condition_loss = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        
        # Tracking
        self.global_step = 0
        self.best_metric = float('inf')
        
        # Initialize logging
        if is_main_process():
            wandb.init(project=config.wandb_project, config=config.__dict__)
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager()
    
    def train(self, train_dataset: Dataset, val_dataset: Dataset, 
              output_dir: str):
        """
        Train the VLM model
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir: Directory to save outputs
        """
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )
        
        # Training loop
        for epoch in range(self.config.epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.config.epochs}")
            
            # Train epoch
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # Validation
            if (epoch + 1) % 5 == 0:  # Validate every 5 epochs
                val_metrics = self._validate(val_loader)
                
                # Save best model
                if val_metrics["total_loss"] < self.best_metric:
                    self.best_metric = val_metrics["total_loss"]
                    self._save_checkpoint(
                        os.path.join(output_dir, "best_model.pth"),
                        is_best=True
                    )
                
                # Log validation metrics
                if is_main_process():
                    wandb.log({
                        f"val/{k}": v for k, v in val_metrics.items()
                    }, step=self.global_step)
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(
                    os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}.pth")
                )
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_metrics = {
            "total_loss": 0.0,
            "navigation_loss": 0.0,
            "condition_loss": 0.0
        }
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}") as pbar:
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                images = {
                    res: imgs.to(self.device) 
                    for res, imgs in batch["images"].items()
                }
                
                # Forward pass
                with autocast(enabled=self.config.mixed_precision):
                    outputs = self.model(images, batch.get("caption"))
                    
                    # Calculate losses
                    total_loss = 0.0
                    
                    # Navigation loss
                    if "navigation_features" in outputs:
                        nav_loss = self._compute_navigation_loss(
                            outputs["navigation_features"],
                            batch["navigation_labels"]
                        )
                        total_loss += nav_loss
                        epoch_metrics["navigation_loss"] += nav_loss.item()
                    
                    # Condition losses
                    if "condition_predictions" in outputs:
                        cond_loss = self._compute_condition_loss(
                            outputs["condition_predictions"],
                            batch
                        )
                        total_loss += cond_loss
                        epoch_metrics["condition_loss"] += cond_loss.item()
                    
                    epoch_metrics["total_loss"] += total_loss.item()
                
                # Backward pass
                if self.scaler:
                    self.scaler.scale(total_loss).backward()
                    
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config.gradient_clip_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                        self.scheduler.step()
                else:
                    total_loss.backward()
                    
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.gradient_clip_norm
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.scheduler.step()
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    "loss": total_loss.item(),
                    "lr": self.scheduler.get_last_lr()[0]
                })
                
                # Log metrics
                if self.global_step % self.config.log_interval == 0:
                    if is_main_process():
                        wandb.log({
                            "train/total_loss": total_loss.item(),
                            "train/learning_rate": self.scheduler.get_last_lr()[0],
                            "train/epoch": epoch
                        }, step=self.global_step)
                
                self.global_step += 1
        
        # Average epoch metrics
        num_batches = len(train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        
        val_metrics = {
            "total_loss": 0.0,
            "navigation_loss": 0.0,
            "condition_loss": 0.0,
            "condition_accuracy": {}
        }
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                images = {
                    res: imgs.to(self.device)
                    for res, imgs in batch["images"].items()
                }
                
                # Forward pass
                outputs = self.model(images, batch.get("caption"))
                
                # Calculate losses
                total_loss = 0.0
                
                # Navigation loss
                if "navigation_features" in outputs:
                    nav_loss = self._compute_navigation_loss(
                        outputs["navigation_features"],
                        batch["navigation_labels"]
                    )
                    total_loss += nav_loss
                    val_metrics["navigation_loss"] += nav_loss.item()
                
                # Condition losses and accuracy
                if "condition_predictions" in outputs:
                    cond_loss, cond_acc = self._compute_condition_metrics(
                        outputs["condition_predictions"],
                        batch
                    )
                    total_loss += cond_loss
                    val_metrics["condition_loss"] += cond_loss.item()
                    
                    for cond_type, acc in cond_acc.items():
                        if cond_type not in val_metrics["condition_accuracy"]:
                            val_metrics["condition_accuracy"][cond_type] = []
                        val_metrics["condition_accuracy"][cond_type].append(acc)
                
                val_metrics["total_loss"] += total_loss.item()
        
        # Average metrics
        num_batches = len(val_loader)
        val_metrics["total_loss"] /= num_batches
        val_metrics["navigation_loss"] /= num_batches
        val_metrics["condition_loss"] /= num_batches
        
        # Average condition accuracies
        for cond_type in val_metrics["condition_accuracy"]:
            val_metrics["condition_accuracy"][cond_type] = np.mean(
                val_metrics["condition_accuracy"][cond_type]
            )
        
        return val_metrics
    
    def _compute_navigation_loss(self, predictions: torch.Tensor, 
                                labels: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute navigation-specific loss"""
        # For now, simple MSE loss on features
        # In practice, this could be more sophisticated
        if "target_features" in labels:
            target = labels["target_features"].to(self.device)
            return self.navigation_loss(predictions, target)
        return torch.tensor(0.0).to(self.device)
    
    def _compute_condition_loss(self, predictions: Dict[str, torch.Tensor],
                               batch: Dict[str, Any]) -> torch.Tensor:
        """Compute condition prediction losses"""
        total_loss = torch.tensor(0.0).to(self.device)
        
        # Map condition names to indices
        condition_mappings = {
            "time": {"day": 0, "dusk": 1, "dawn": 1, "night": 2},
            "weather": {"clear": 0, "cloudy": 1, "rain": 2, "snow": 3, "fog": 4},
            "season": {"spring": 0, "summer": 1, "fall": 2, "autumn": 2, "winter": 3}
        }
        
        for cond_type, pred in predictions.items():
            if cond_type in batch:
                # Convert string labels to indices
                labels = []
                for cond in batch[cond_type]:
                    if cond in condition_mappings[cond_type]:
                        labels.append(condition_mappings[cond_type][cond])
                    else:
                        labels.append(0)  # Default
                
                labels = torch.tensor(labels).to(self.device)
                loss = self.condition_loss(pred, labels)
                total_loss += loss
        
        return total_loss
    
    def _compute_condition_metrics(self, predictions: Dict[str, torch.Tensor],
                                  batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute condition losses and accuracies"""
        total_loss = torch.tensor(0.0).to(self.device)
        accuracies = {}
        
        condition_mappings = {
            "time": {"day": 0, "dusk": 1, "dawn": 1, "night": 2},
            "weather": {"clear": 0, "cloudy": 1, "rain": 2, "snow": 3, "fog": 4},
            "season": {"spring": 0, "summer": 1, "fall": 2, "autumn": 2, "winter": 3}
        }
        
        for cond_type, pred in predictions.items():
            if cond_type in batch:
                # Convert string labels to indices
                labels = []
                for cond in batch[cond_type]:
                    if cond in condition_mappings[cond_type]:
                        labels.append(condition_mappings[cond_type][cond])
                    else:
                        labels.append(0)
                
                labels = torch.tensor(labels).to(self.device)
                
                # Loss
                loss = self.condition_loss(pred, labels)
                total_loss += loss
                
                # Accuracy
                pred_labels = pred.argmax(dim=-1)
                accuracy = (pred_labels == labels).float().mean().item()
                accuracies[cond_type] = accuracy
        
        return total_loss, accuracies
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer"""
        # Separate parameters for different learning rates
        base_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "navigation_head" in name or "condition_heads" in name:
                    head_params.append(param)
                else:
                    base_params.append(param)
        
        # Use different learning rates
        param_groups = [
            {"params": base_params, "lr": self.config.learning_rate * 0.1},
            {"params": head_params, "lr": self.config.learning_rate}
        ]
        
        return optim.AdamW(
            param_groups,
            weight_decay=self.config.weight_decay
        )
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        from transformers import get_linear_schedule_with_warmup
        
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=100000  # Will be updated based on dataset
        )
    
    def _save_checkpoint(self, path: str, is_best: bool = False):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "global_step": self.global_step,
            "best_metric": self.best_metric
        }
        
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
        
        if is_best:
            logger.info(f"New best model! Metric: {self.best_metric:.4f}")


def main():
    """Main training function"""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Train VLM for navigation")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--data_path", type=str, required=True, help="Dataset path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = VLMTrainingConfig(**config_dict.get("vlm", {}))
    
    # Setup distributed training if available
    setup_distributed()
    
    # Create datasets
    train_dataset = NavigationVLMDataset(args.data_path, config, split="train")
    val_dataset = NavigationVLMDataset(args.data_path, config, split="val")
    
    # Create trainer
    trainer = VLMTrainer(config)
    
    # Resume if specified
    if args.resume:
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        trainer.global_step = checkpoint["global_step"]
        trainer.best_metric = checkpoint["best_metric"]
        logger.info(f"Resumed from checkpoint: {args.resume}")
    
    # Train
    trainer.train(train_dataset, val_dataset, args.output_dir)


if __name__ == "__main__":
    main()
