"""
Configuration settings for spatialLM training and evaluation.

This file contains default configuration settings that can be imported and
used throughout the project.
"""

import os
from typing import Dict, Any, List, Union, Optional
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    """Configuration for model architecture"""
    base_model_type: str = "gpt2"
    base_model_name: str = "gpt2"
    spatial_dim: int = 3
    spatial_hidden_size: int = 256
    spatial_dropout: float = 0.1
    spatial_norm_eps: float = 1e-12
    use_spatial_embeddings: bool = True
    spatial_embedding_size: int = 64
    use_spatial_attention: bool = True
    spatial_attention_heads: int = 4
    spatial_context_size: int = 16
    max_sequence_length: int = 512

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    output_dir: str = "./trained_models"
    overwrite_output_dir: bool = False
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    num_train_epochs: float = 3.0
    max_steps: int = -1
    warmup_ratio: float = 0.0
    logging_steps: int = 500
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    fp16: bool = False
    bf16: bool = False
    seed: int = 42
    data_seed: int = 42
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_token: Optional[str] = None
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.0
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])

@dataclass
class FinetuningConfig:
    """Configuration for model fine-tuning"""
    finetune_method: str = "full"  # full, lora, qlora, peft
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bits: int = 4  # For quantization (4 or 8)

@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing"""
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    text_column: str = "text"
    spatial_mode: bool = False
    coordinate_columns: List[str] = field(default_factory=lambda: ["x", "y", "z"])
    data_augmentation: bool = False
    augmentation_factor: int = 2
    validation_split_percentage: int = 10
    preprocessing_num_workers: Optional[int] = None
    overwrite_cache: bool = False

@dataclass
class DeploymentConfig:
    """Configuration for model deployment"""
    quantize: bool = False
    quantization_type: str = "dynamic"  # dynamic, float16, int8
    optimize_for: str = "default"  # default, storage, latency
    include_tokenizer: bool = True
    minimum_deployment_target: str = "iOS15"  # For iOS
    compute_units: str = "ALL"  # ALL, CPU_ONLY, CPU_AND_GPU, CPU_AND_NE (for iOS)
    precision: str = "float32"  # float32, float16 (for iOS)

@dataclass
class SpatialLMConfig:
    """Full configuration for spatialLM"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    finetuning: FinetuningConfig = field(default_factory=FinetuningConfig)
    data: DataConfig = field(default_factory=DataConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "model": {k: v for k, v in self.model.__dict__.items()},
            "training": {k: v for k, v in self.training.__dict__.items()},
            "finetuning": {k: v for k, v in self.finetuning.__dict__.items()},
            "data": {k: v for k, v in self.data.__dict__.items()},
            "deployment": {k: v for k, v in self.deployment.__dict__.items()},
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SpatialLMConfig":
        """Create config from dictionary"""
        model_config = ModelConfig(**config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        finetuning_config = FinetuningConfig(**config_dict.get("finetuning", {}))
        data_config = DataConfig(**config_dict.get("data", {}))
        deployment_config = DeploymentConfig(**config_dict.get("deployment", {}))
        
        return cls(
            model=model_config,
            training=training_config,
            finetuning=finetuning_config,
            data=data_config,
            deployment=deployment_config
        )

    def save(self, config_path: str) -> None:
        """Save config to JSON file"""
        import json
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, config_path: str) -> "SpatialLMConfig":
        """Load config from JSON file"""
        import json
        
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)


# Default configurations for different scenarios
DEFAULT_CONFIG = SpatialLMConfig()

SMALL_MODEL_CONFIG = SpatialLMConfig(
    model=ModelConfig(
        base_model_name="gpt2",
        spatial_hidden_size=128,
        spatial_embedding_size=32,
        spatial_attention_heads=2
    )
)

LARGE_MODEL_CONFIG = SpatialLMConfig(
    model=ModelConfig(
        base_model_name="gpt2-large",
        spatial_hidden_size=512,
        spatial_embedding_size=128,
        spatial_attention_heads=8
    )
)

MOBILE_DEPLOYMENT_CONFIG = SpatialLMConfig(
    model=ModelConfig(
        base_model_name="gpt2",
        spatial_hidden_size=128,
        spatial_embedding_size=32,
        spatial_attention_heads=2,
        max_sequence_length=128
    ),
    deployment=DeploymentConfig(
        quantize=True,
        quantization_type="dynamic",
        optimize_for="latency",
        include_tokenizer=True
    )
)
