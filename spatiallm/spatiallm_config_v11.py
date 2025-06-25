#!/usr/bin/env python3
"""
Configuration settings for spatialLM v1.1 training and evaluation.

This file contains updated configuration settings for version 1.1 with enhanced
spatial reasoning capabilities, improved mobile optimization, and better
deployment options.

Version 1.1 Updates:
- Enhanced spatial attention mechanisms
- Improved quantization-aware training
- Better mobile deployment configurations
- Advanced optimization settings
"""

import os
from typing import Dict, Any, List, Union, Optional
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    """Configuration for model architecture - Updated for v1.1 with Qwen support"""
    base_model_type: str = "qwen"  # Updated default to qwen
    base_model_name: str = "Qwen/Qwen-0.5B"  # Updated default to Qwen
    
    # Spatial configuration - Enhanced in v1.1
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
    
    # New v1.1 features
    enhanced_spatial_reasoning: bool = True
    spatial_fusion_method: str = "cross_attention"  # cross_attention, concatenation, gating
    spatial_positional_encoding: bool = True
    spatial_coordinate_normalization: str = "layer_norm"  # layer_norm, batch_norm, none
    spatial_attention_temperature: float = 1.0
    
    # Qwen-specific configurations
    qwen_architecture: bool = True
    qwen_vocab_size: int = 151936  # Qwen-0.5B vocab size
    qwen_hidden_size: int = 1024   # Qwen-0.5B hidden size
    qwen_num_layers: int = 24      # Qwen-0.5B layers
    qwen_num_attention_heads: int = 16  # Qwen-0.5B attention heads
    qwen_intermediate_size: int = 2816  # Qwen-0.5B intermediate size
    
    # Multi-scale spatial processing
    multi_scale_spatial: bool = True
    spatial_scales: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25])
    spatial_scale_fusion: str = "weighted_sum"  # weighted_sum, concatenation, attention
    
    # Advanced spatial features
    spatial_memory_bank: bool = False
    spatial_memory_size: int = 1024
    dynamic_spatial_attention: bool = True
    spatial_relation_modeling: bool = True
    
    # Mobile optimization features
    mobile_optimized: bool = False
    quantization_aware_training: bool = False
    pruning_aware_training: bool = False
    knowledge_distillation: bool = False
    
    # Model efficiency
    gradient_checkpointing: bool = False
    use_flash_attention: bool = False
    attention_implementation: str = "default"  # default, flash, linear, sparse

@dataclass
class TrainingConfig:
    """Configuration for model training - Enhanced for v1.1"""
    output_dir: str = "./trained_models_v1.1"
    overwrite_output_dir: bool = False
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_train_epochs: float = 3.0
    max_steps: int = -1
    warmup_ratio: float = 0.1
    logging_steps: int = 500
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    fp16: bool = False
    bf16: bool = True  # Preferred for v1.1
    seed: int = 42
    data_seed: int = 42
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_token: Optional[str] = None
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001
    report_to: List[str] = field(default_factory=lambda: ["tensorboard", "wandb"])
    
    # Advanced training features for v1.1
    use_AdamW: bool = True
    optimizer_type: str = "adamw"  # adamw, lion, adafactor
    scheduler_type: str = "cosine"  # linear, cosine, polynomial, constant
    cosine_restarts: bool = False
    
    # Spatial-specific training
    spatial_loss_weight: float = 1.0
    coordinate_loss_type: str = "mse"  # mse, l1, huber
    spatial_consistency_loss: bool = True
    spatial_consistency_weight: float = 0.1
    
    # Regularization
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    spatial_dropout: float = 0.1
    layer_norm_eps: float = 1e-12
    
    # Advanced optimization
    gradient_clipping_type: str = "norm"  # norm, value
    use_gradient_accumulation: bool = True
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    
    # Checkpointing and resuming
    resume_from_checkpoint: Optional[str] = None
    auto_find_batch_size: bool = False
    include_inputs_for_metrics: bool = False

@dataclass
class FinetuningConfig:
    """Configuration for model fine-tuning - Enhanced for v1.1"""
    finetune_method: str = "lora"  # full, lora, qlora, peft, adalora
    
    # LoRA configuration
    lora_rank: int = 16  # Increased for v1.1
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])
    lora_bias: str = "none"  # none, all, lora_only
    
    # AdaLoRA configuration
    adalora_target_r: int = 8
    adalora_init_r: int = 12
    adalora_tinit: int = 200
    adalora_tfinal: int = 1000
    adalora_deltaT: int = 10
    
    # QLoRA configuration
    bits: int = 4  # For quantization (4 or 8)
    quant_type: str = "nf4"  # nf4, fp4
    use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    
    # Spatial-specific fine-tuning
    freeze_base_model: bool = False
    freeze_spatial_layers: bool = False
    spatial_layer_learning_rate: float = 1e-4
    
    # Task-specific fine-tuning
    task_type: str = "spatial_reasoning"  # spatial_reasoning, navigation, object_detection
    use_task_specific_head: bool = True
    head_hidden_size: int = 256
    head_num_layers: int = 2

@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing - Enhanced for v1.1"""
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    
    # Text and spatial configuration
    text_column: str = "text"
    spatial_mode: bool = True  # Default to True for v1.1
    coordinate_columns: List[str] = field(default_factory=lambda: ["x", "y", "z"])
    coordinate_format: str = "float"  # float, normalized, categorical
    
    # Enhanced spatial data processing
    spatial_coordinate_system: str = "cartesian"  # cartesian, polar, spherical, geographic
    coordinate_normalization: bool = True
    coordinate_bounds: Optional[Dict[str, tuple]] = None  # {"x": (-100, 100), "y": (-100, 100)}
    relative_coordinates: bool = False
    spatial_reference_point: Optional[List[float]] = None
    
    # Data augmentation - Enhanced for v1.1
    data_augmentation: bool = True
    augmentation_factor: int = 3  # Increased for v1.1
    spatial_augmentation: bool = True
    coordinate_noise_std: float = 0.1
    rotation_augmentation: bool = True
    scale_augmentation: bool = True
    translation_augmentation: bool = True
    
    # Text augmentation
    text_augmentation: bool = True
    synonym_replacement: bool = True
    random_insertion: bool = True
    random_swap: bool = True
    random_deletion: bool = True
    augmentation_probability: float = 0.1
    
    # Data validation and preprocessing
    validation_split_percentage: int = 10
    test_split_percentage: int = 10
    preprocessing_num_workers: Optional[int] = None
    overwrite_cache: bool = False
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    
    # Advanced data features
    streaming: bool = False
    cache_file_name: Optional[str] = None
    keep_in_memory: bool = False
    save_intermediate_datasets: bool = True

@dataclass
class DeploymentConfig:
    """Configuration for model deployment - Significantly enhanced for v1.1"""
    
    # Basic deployment settings
    quantize: bool = True
    quantization_type: str = "int8"  # dynamic, float16, int8, int4
    optimize_for: str = "mobile"  # default, storage, latency, mobile, edge
    include_tokenizer: bool = True
    
    # Mobile-specific settings
    mobile_deployment: bool = True
    target_platforms: List[str] = field(default_factory=lambda: ["ios", "android"])
    
    # iOS deployment configuration
    ios_config: Dict[str, Any] = field(default_factory=lambda: {
        "minimum_deployment_target": "iOS16",
        "compute_units": "CPU_AND_NE",
        "precision": "float16",
        "quantization_mode": "linear_symmetric",
        "use_palettization": False,
        "neural_engine_optimization": True,
        "model_format": "mlpackage"
    })
    
    # Android deployment configuration
    android_config: Dict[str, Any] = field(default_factory=lambda: {
        "min_sdk_version": 24,
        "target_sdk_version": 34,
        "quantization_mode": "int8",
        "use_gpu_delegate": True,
        "use_nnapi_delegate": True,
        "use_xnnpack": True,
        "optimization_profile": "balanced"
    })
    
    # Model optimization
    pruning: bool = False
    pruning_ratio: float = 0.2
    structured_pruning: bool = True
    knowledge_distillation: bool = False
    teacher_model: Optional[str] = None
    distillation_temperature: float = 4.0
    distillation_alpha: float = 0.7
    
    # Advanced deployment features
    batch_size_optimization: bool = True
    sequence_length_optimization: bool = True
    memory_optimization: bool = True
    inference_optimization: bool = True
    
    # Edge deployment
    edge_deployment: bool = False
    edge_target: str = "raspberry_pi"  # raspberry_pi, jetson_nano, coral_tpu
    edge_precision: str = "int8"
    
    # Validation and testing
    validate_converted_model: bool = True
    performance_benchmarking: bool = True
    accuracy_threshold: float = 0.95
    latency_threshold_ms: float = 200
    memory_threshold_mb: float = 1000

@dataclass
class SpatialLMConfig:
    """Full configuration for spatialLM v1.1"""
    version: str = "1.1"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    finetuning: FinetuningConfig = field(default_factory=FinetuningConfig)
    data: DataConfig = field(default_factory=DataConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "version": self.version,
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
            version=config_dict.get("version", "1.1"),
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

# Predefined configurations for different use cases

DEFAULT_V1_1_CONFIG = SpatialLMConfig()

SMALL_MOBILE_CONFIG = SpatialLMConfig(
    model=ModelConfig(
        base_model_name="manycore-research/SpatialLM1.1-Qwen-0.5B",
        base_model_type="qwen",
        qwen_architecture=True,
        spatial_hidden_size=128,
        spatial_embedding_size=32,
        spatial_attention_heads=2,
        mobile_optimized=True,
        quantization_aware_training=True,
        max_sequence_length=128
    ),
    training=TrainingConfig(
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
        num_train_epochs=5.0,
        bf16=True
    ),
    finetuning=FinetuningConfig(
        finetune_method="lora",
        lora_rank=8,
        lora_alpha=16,
        bits=8
    ),
    deployment=DeploymentConfig(
        quantize=True,
        quantization_type="int8",
        mobile_deployment=True,
        pruning=True,
        pruning_ratio=0.3
    )
)

LARGE_SERVER_CONFIG = SpatialLMConfig(
    model=ModelConfig(
        base_model_name="gpt2-large",
        spatial_hidden_size=512,
        spatial_embedding_size=128,
        spatial_attention_heads=8,
        enhanced_spatial_reasoning=True,
        multi_scale_spatial=True,
        spatial_memory_bank=True,
        use_flash_attention=True
    ),
    training=TrainingConfig(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        num_train_epochs=10.0,
        bf16=True,
        gradient_checkpointing=True
    ),
    finetuning=FinetuningConfig(
        finetune_method="lora",
        lora_rank=32,
        lora_alpha=64,
        lora_target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj",
            "spatial_attention.q_proj", "spatial_attention.v_proj"
        ]
    ),
    deployment=DeploymentConfig(
        quantize=True,
        quantization_type="int4",
        mobile_deployment=False,
        knowledge_distillation=True
    )
)

EDGE_DEPLOYMENT_CONFIG = SpatialLMConfig(
    model=ModelConfig(
        base_model_name="distilgpt2",
        spatial_hidden_size=96,
        spatial_embedding_size=24,
        spatial_attention_heads=2,
        mobile_optimized=True,
        quantization_aware_training=True,
        max_sequence_length=64
    ),
    training=TrainingConfig(
        per_device_train_batch_size=32,
        learning_rate=5e-4,
        num_train_epochs=3.0,
        bf16=True
    ),
    finetuning=FinetuningConfig(
        finetune_method="qlora",
        lora_rank=4,
        lora_alpha=8,
        bits=4
    ),
    deployment=DeploymentConfig(
        quantize=True,
        quantization_type="int8",
        mobile_deployment=True,
        edge_deployment=True,
        pruning=True,
        pruning_ratio=0.5,
        knowledge_distillation=True
    )
)

NAVIGATION_TASK_CONFIG = SpatialLMConfig(
    model=ModelConfig(
        base_model_name="gpt2",
        spatial_hidden_size=256,
        spatial_embedding_size=64,
        spatial_attention_heads=4,
        enhanced_spatial_reasoning=True,
        spatial_relation_modeling=True,
        dynamic_spatial_attention=True
    ),
    data=DataConfig(
        spatial_mode=True,
        coordinate_columns=["lat", "lon", "altitude"],
        spatial_coordinate_system="geographic",
        coordinate_normalization=True,
        spatial_augmentation=True,
        data_augmentation=True
    ),
    finetuning=FinetuningConfig(
        task_type="navigation",
        use_task_specific_head=True,
        head_hidden_size=128
    )
)

# Utility functions for configuration management

def get_config_for_task(task: str) -> SpatialLMConfig:
    """Get predefined configuration for specific tasks"""
    task_configs = {
        "mobile": SMALL_MOBILE_CONFIG,
        "server": LARGE_SERVER_CONFIG,
        "edge": EDGE_DEPLOYMENT_CONFIG,
        "navigation": NAVIGATION_TASK_CONFIG,
        "default": DEFAULT_V1_1_CONFIG
    }
    
    return task_configs.get(task, DEFAULT_V1_1_CONFIG)

def create_custom_config(
    model_size: str = "base",
    deployment_target: str = "mobile",
    task: str = "general",
    quantization: bool = True
) -> SpatialLMConfig:
    """Create a custom configuration based on requirements"""
    
    # Start with default config
    config = SpatialLMConfig()
    
    # Adjust model size
    if model_size == "small":
        config.model.base_model_name = "gpt2"
        config.model.spatial_hidden_size = 128
        config.model.spatial_embedding_size = 32
        config.model.spatial_attention_heads = 2
    elif model_size == "large":
        config.model.base_model_name = "gpt2-large"
        config.model.spatial_hidden_size = 512
        config.model.spatial_embedding_size = 128
        config.model.spatial_attention_heads = 8
    
    # Adjust for deployment target
    if deployment_target == "mobile":
        config.model.mobile_optimized = True
        config.model.quantization_aware_training = True
        config.deployment.mobile_deployment = True
        config.deployment.quantize = True
    elif deployment_target == "edge":
        config.deployment.edge_deployment = True
        config.deployment.pruning = True
        config.deployment.pruning_ratio = 0.5
    
    # Adjust for task
    if task == "navigation":
        config.data.coordinate_columns = ["lat", "lon", "altitude"]
        config.data.spatial_coordinate_system = "geographic"
        config.model.spatial_relation_modeling = True
    
    # Apply quantization settings
    if quantization:
        config.deployment.quantize = True
        config.model.quantization_aware_training = True
    
    return config

def validate_config(config: SpatialLMConfig) -> List[str]:
    """Validate configuration and return list of issues"""
    issues = []
    
    # Check version compatibility
    if config.version != "1.1":
        issues.append(f"Config version {config.version} may not be compatible with v1.1 features")
    
    # Check model configuration
    if config.model.spatial_attention_heads > 8:
        issues.append("Large number of spatial attention heads may impact performance")
    
    if config.model.mobile_optimized and config.model.spatial_hidden_size > 256:
        issues.append("Large spatial hidden size may not be suitable for mobile deployment")
    
    # Check training configuration
    if config.training.bf16 and config.training.fp16:
        issues.append("Both bf16 and fp16 enabled - only one should be used")
    
    # Check deployment configuration
    if config.deployment.mobile_deployment and not config.deployment.quantize:
        issues.append("Mobile deployment recommended with quantization enabled")
    
    if config.deployment.edge_deployment and config.model.spatial_hidden_size > 128:
        issues.append("Large model may not be suitable for edge deployment")
    
    return issues