#!/usr/bin/env python3
"""
TinyLlama Mobile Training and Deployment Setup
Creates complete directory structure and scripts for training and deploying TinyLlama to mobile platforms.
"""

import os
import sys
from pathlib import Path

def create_directory_structure():
    """Create the complete directory structure for TinyLlama mobile deployment."""
    
    base_dirs = [
        "tinyllama_mobile",
        "tinyllama_mobile/config",
        "tinyllama_mobile/data",
        "tinyllama_mobile/data/raw",
        "tinyllama_mobile/data/processed",
        "tinyllama_mobile/data/tokenizer",
        "tinyllama_mobile/models",
        "tinyllama_mobile/models/pretrained",
        "tinyllama_mobile/models/finetuned", 
        "tinyllama_mobile/models/mobile",
        "tinyllama_mobile/models/mobile/ios",
        "tinyllama_mobile/models/mobile/android",
        "tinyllama_mobile/src",
        "tinyllama_mobile/scripts",
        "tinyllama_mobile/training",
        "tinyllama_mobile/deployment",
        "tinyllama_mobile/deployment/ios",
        "tinyllama_mobile/deployment/android",
        "tinyllama_mobile/tests",
        "tinyllama_mobile/notebooks",
        "tinyllama_mobile/examples"
    ]
    
    for dir_path in base_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

def create_requirements_txt():
    """Create requirements.txt file."""
    requirements = """# Core ML/AI packages
torch>=2.0.0
transformers>=4.35.0
tokenizers>=0.15.0
datasets>=2.14.0
accelerate>=0.20.0
peft>=0.6.0

# Training utilities
sentencepiece>=0.1.99
safetensors>=0.3.0
wandb>=0.15.0
tensorboard>=2.13.0
scikit-learn>=1.3.0

# Mobile deployment
onnx>=1.14.0
onnxruntime>=1.15.0
coremltools>=7.0
tensorflow>=2.13.0
onnx-tf>=1.10.0

# Data processing
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
pyyaml>=6.0

# Utilities
huggingface_hub>=0.17.0
jupyter>=1.0.0
ipykernel>=6.25.0
requests>=2.31.0
pillow>=10.0.0

# Development
pytest>=7.4.0
black>=23.7.0
flake8>=6.0.0
mypy>=1.5.0
"""
    
    with open("tinyllama_mobile/requirements.txt", "w") as f:
        f.write(requirements)
    print("Created requirements.txt")

def create_main_config():
    """Create main configuration file."""
    config = """# TinyLlama Mobile Configuration

model:
  name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  cache_dir: "./models/pretrained/"
  
  # Model parameters
  vocab_size: 32000
  hidden_size: 2048
  intermediate_size: 5632
  num_hidden_layers: 22
  num_attention_heads: 32
  max_position_embeddings: 2048
  
tokenizer:
  type: "sentencepiece"
  vocab_size: 32000
  model_max_length: 2048
  padding_side: "left"
  
  # Custom vocabulary training
  custom_vocab:
    enabled: false
    corpus_path: "./data/raw/training_corpus.txt"
    vocab_path: "./data/tokenizer/"
    
training:
  # Basic parameters
  num_epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 2e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  
  # LoRA configuration
  lora:
    enabled: true
    r: 8
    alpha: 16
    dropout: 0.1
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
  # QLoRA configuration  
  qlora:
    enabled: false
    bits: 4
    quant_type: "nf4"
    use_double_quant: true
    
mobile:
  # iOS deployment
  ios:
    target_version: "15.0"
    compute_units: "neural_engine"
    precision: "float16"
    quantization:
      enabled: true
      bits: 8
      
  # Android deployment  
  android:
    target_api: 24
    delegates: ["gpu", "nnapi"]
    quantization:
      enabled: true
      representative_dataset_size: 100
      
optimization:
  # Model pruning
  pruning:
    enabled: false
    sparsity: 0.1
    
  # Knowledge distillation
  distillation:
    enabled: false
    teacher_model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    temperature: 4.0
    alpha: 0.7
"""
    
    with open("tinyllama_mobile/config/config.yaml", "w") as f:
        f.write(config)
    print("Created config/config.yaml")

def create_tokenizer_trainer():
    """Create vocabulary/tokenizer training script."""
    script = '''#!/usr/bin/env python3
"""
TinyLlama Tokenizer Trainer
Trains a custom SentencePiece tokenizer for domain-specific vocabulary.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional
import sentencepiece as spm
import yaml
from transformers import AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TokenizerTrainer:
    """Trains custom tokenizers for TinyLlama."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.tokenizer_config = self.config['tokenizer']
        
    def prepare_training_corpus(self, 
                               input_files: List[str], 
                               output_file: str,
                               max_sentences: Optional[int] = None) -> None:
        """Prepare training corpus from multiple input files."""
        logger.info(f"Preparing training corpus from {len(input_files)} files")
        
        sentence_count = 0
        
        with open(output_file, 'w', encoding='utf-8') as outf:
            for input_file in input_files:
                logger.info(f"Processing {input_file}")
                
                try:
                    with open(input_file, 'r', encoding='utf-8') as inf:
                        for line in inf:
                            line = line.strip()
                            if line:
                                outf.write(line + '\\n')
                                sentence_count += 1
                                
                                if max_sentences and sentence_count >= max_sentences:
                                    logger.info(f"Reached max sentences limit: {max_sentences}")
                                    return
                                    
                except Exception as e:
                    logger.error(f"Error processing {input_file}: {e}")
                    
        logger.info(f"Prepared corpus with {sentence_count} sentences")
        
    def train_sentencepiece_tokenizer(self, 
                                    corpus_file: str,
                                    output_dir: str,
                                    vocab_size: int = 32000) -> str:
        """Train SentencePiece tokenizer."""
        logger.info(f"Training SentencePiece tokenizer with vocab size {vocab_size}")
        
        os.makedirs(output_dir, exist_ok=True)
        model_prefix = os.path.join(output_dir, "tinyllama_tokenizer")
        
        # SentencePiece training arguments
        spm_args = [
            f"--input={corpus_file}",
            f"--model_prefix={model_prefix}",
            f"--vocab_size={vocab_size}",
            "--model_type=bpe",
            "--character_coverage=0.9995",
            "--num_threads=16",
            "--split_digits=true",
            "--allow_whitespace_only_pieces=true",
            "--byte_fallback=true",
            "--normalization_rule_name=identity",
            "--remove_extra_whitespaces=false",
            "--input_sentence_size=10000000",
            "--max_sentence_length=8192",
            # Special tokens
            "--pad_id=0",
            "--unk_id=1", 
            "--bos_id=2",
            "--eos_id=3",
            "--user_defined_symbols=<|im_start|>,<|im_end|>,<|system|>,<|user|>,<|assistant|>"
        ]
        
        # Train tokenizer
        spm.SentencePieceTrainer.train(" ".join(spm_args))
        
        model_file = f"{model_prefix}.model"
        vocab_file = f"{model_prefix}.vocab"
        
        logger.info(f"Tokenizer saved to {model_file}")
        return model_file
        
    def create_hf_tokenizer(self, 
                           spm_model_path: str, 
                           output_dir: str) -> None:
        """Create HuggingFace compatible tokenizer."""
        logger.info("Creating HuggingFace compatible tokenizer")
        
        # Load the original TinyLlama tokenizer as template
        base_tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['name'],
            cache_dir=self.config['model']['cache_dir']
        )
        
        # Create new tokenizer with custom vocabulary
        from transformers import LlamaTokenizer
        
        tokenizer = LlamaTokenizer(
            vocab_file=smp_model_path,
            add_bos_token=True,
            add_eos_token=True,
            clean_up_tokenization_spaces=False,
            pad_token="<pad>",
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
        )
        
        # Add special tokens for chat format
        special_tokens = {
            "additional_special_tokens": [
                "<|im_start|>", "<|im_end|>", 
                "<|system|>", "<|user|>", "<|assistant|>"
            ]
        }
        
        tokenizer.add_special_tokens(special_tokens)
        
        # Save tokenizer
        tokenizer.save_pretrained(output_dir)
        logger.info(f"HuggingFace tokenizer saved to {output_dir}")
        
    def validate_tokenizer(self, tokenizer_path: str) -> None:
        """Validate the trained tokenizer."""
        logger.info("Validating tokenizer")
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Test sentences
        test_sentences = [
            "Hello, how are you today?",
            "The quick brown fox jumps over the lazy dog.",
            "What is machine learning?",
            "I love artificial intelligence and natural language processing.",
            "<|im_start|>system\\nYou are a helpful assistant.<|im_end|>"
        ]
        
        for sentence in test_sentences:
            tokens = tokenizer.tokenize(sentence)
            ids = tokenizer.encode(sentence)
            decoded = tokenizer.decode(ids)
            
            print(f"Original: {sentence}")
            print(f"Tokens: {tokens}")
            print(f"IDs: {ids}")
            print(f"Decoded: {decoded}")
            print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="Train custom tokenizer for TinyLlama")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--corpus_files", type=str, nargs="+", required=True,
                       help="Path to training corpus files")
    parser.add_argument("--output_dir", type=str, default="data/tokenizer/",
                       help="Output directory for tokenizer")
    parser.add_argument("--vocab_size", type=int, default=32000,
                       help="Vocabulary size")
    parser.add_argument("--max_sentences", type=int, default=None,
                       help="Maximum number of sentences to use")
    parser.add_argument("--validate", action="store_true",
                       help="Validate the trained tokenizer")
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = TokenizerTrainer(args.config)
        
        # Prepare corpus
        corpus_file = os.path.join(args.output_dir, "training_corpus.txt")
        os.makedirs(args.output_dir, exist_ok=True)
        
        trainer.prepare_training_corpus(
            args.corpus_files, 
            corpus_file,
            args.max_sentences
        )
        
        # Train SentencePiece tokenizer
        smp_model_path = trainer.train_sentencepiece_tokenizer(
            corpus_file,
            args.output_dir, 
            args.vocab_size
        )
        
        # Create HuggingFace tokenizer
        hf_tokenizer_dir = os.path.join(args.output_dir, "hf_tokenizer")
        trainer.create_hf_tokenizer(smp_model_path, hf_tokenizer_dir)
        
        # Validate if requested
        if args.validate:
            trainer.validate_tokenizer(hf_tokenizer_dir)
            
        print("Tokenizer training completed successfully!")
        print(f"Tokenizer saved to: {hf_tokenizer_dir}")
        
    except Exception as e:
        logger.error(f"Error training tokenizer: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    with open("tinyllama_mobile/scripts/train_tokenizer.py", "w") as f:
        f.write(script)
    os.chmod("tinyllama_mobile/scripts/train_tokenizer.py", 0o755)
    print("Created scripts/train_tokenizer.py")

def create_model_trainer():
    """Create main model training script."""
    script = '''#!/usr/bin/env python3
"""
TinyLlama Model Training Script
Fine-tunes TinyLlama for specific tasks with LoRA/QLoRA support.
"""

import argparse
import os
import sys
import torch
import yaml
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    EarlyStoppingCallback, get_scheduler
)
from peft import (
    LoraConfig, get_peft_model, TaskType, 
    prepare_model_for_kbit_training
)
from datasets import load_dataset, Dataset
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TinyLlamaTrainer:
    """TinyLlama model trainer with LoRA/QLoRA support."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        
    def setup_model_and_tokenizer(self, custom_tokenizer_path: Optional[str] = None):
        """Setup model and tokenizer."""
        logger.info("Setting up model and tokenizer...")
        
        model_name = self.config['model']['name']
        
        # Load tokenizer
        if custom_tokenizer_path and os.path.exists(custom_tokenizer_path):
            logger.info(f"Loading custom tokenizer from {custom_tokenizer_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(custom_tokenizer_path)
        else:
            logger.info(f"Loading default tokenizer for {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model
        if self.config['training']['qlora']['enabled']:
            # QLoRA setup
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config['training']['qlora']['quant_type'],
                bnb_4bit_use_double_quant=self.config['training']['qlora']['use_double_quant'],
                bnb_4bit_compute_dtype=torch.float16
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            self.model = prepare_model_for_kbit_training(self.model)
            
        else:
            # Standard loading
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
        
        # Setup LoRA if enabled
        if self.config['training']['lora']['enabled']:
            self.setup_lora()
            
        # Resize token embeddings if using custom tokenizer
        if custom_tokenizer_path:
            self.model.resize_token_embeddings(len(self.tokenizer))
            
    def setup_lora(self):
        """Setup LoRA configuration."""
        logger.info("Setting up LoRA...")
        
        lora_config = LoraConfig(
            r=self.config['training']['lora']['r'],
            lora_alpha=self.config['training']['lora']['alpha'],
            target_modules=self.config['training']['lora']['target_modules'],
            lora_dropout=self.config['training']['lora']['dropout'],
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
    def prepare_dataset(self, data_path: str, max_length: int = 512):
        """Prepare training dataset."""
        logger.info(f"Preparing dataset from {data_path}")
        
        # Load dataset
        if data_path.endswith('.json') or data_path.endswith('.jsonl'):
            dataset = load_dataset('json', data_files=data_path)['train']
        elif data_path.endswith('.csv'):
            dataset = load_dataset('csv', data_files=data_path)['train']
        elif data_path.endswith('.txt'):
            dataset = load_dataset('text', data_files=data_path)['train']
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # Tokenize dataset
        def tokenize_function(examples):
            # Assuming the dataset has a 'text' column
            text_column = 'text' if 'text' in examples else list(examples.keys())[0]
            
            # Format for chat if needed
            texts = []
            for text in examples[text_column]:
                # Add chat format if not already present
                if not text.startswith('<|im_start|>'):
                    formatted_text = f"<|im_start|>user\\n{text}<|im_end|>\\n<|im_start|>assistant\\n"
                    texts.append(formatted_text)
                else:
                    texts.append(text)
            
            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=max_length,
                return_tensors=None
            )
            
            # Create labels (same as input_ids for causal LM)
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
        
    def train(self, 
              train_dataset,
              eval_dataset=None,
              output_dir: str = "models/finetuned",
              run_name: Optional[str] = None):
        """Train the model."""
        logger.info("Starting training...")
        
        # Create output directory
        if run_name is None:
            run_name = f"tinyllama_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        full_output_dir = os.path.join(output_dir, run_name)
        os.makedirs(full_output_dir, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=full_output_dir,
            num_train_epochs=self.config['training']['num_epochs'],
            per_device_train_batch_size=self.config['training']['batch_size'],
            per_device_eval_batch_size=self.config['training']['batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            learning_rate=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            warmup_ratio=self.config['training']['warmup_ratio'],
            logging_steps=50,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=200 if eval_dataset else None,
            save_steps=500,
            save_total_limit=3,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to="wandb" if wandb.api.api_key else "none",
            run_name=run_name,
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Callbacks
        callbacks = []
        if eval_dataset:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
        )
        
        # Start training
        trainer.train()
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(full_output_dir)
        
        # Save training config
        with open(os.path.join(full_output_dir, "training_config.json"), "w") as f:
            json.dump(self.config, f, indent=2)
            
        logger.info(f"Training completed. Model saved to {full_output_dir}")
        return full_output_dir

def main():
    parser = argparse.ArgumentParser(description="Train TinyLlama model")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--train_data", type=str, required=True,
                       help="Path to training data")
    parser.add_argument("--eval_data", type=str, default=None,
                       help="Path to evaluation data")
    parser.add_argument("--custom_tokenizer", type=str, default=None,
                       help="Path to custom tokenizer")
    parser.add_argument("--output_dir", type=str, default="models/finetuned",
                       help="Output directory")
    parser.add_argument("--run_name", type=str, default=None,
                       help="Run name for this training")
    parser.add_argument("--wandb_project", type=str, default="tinyllama-mobile",
                       help="Weights & Biases project name")
    
    args = parser.parse_args()
    
    try:
        # Initialize W&B if available
        if wandb.api.api_key:
            wandb.init(project=args.wandb_project)
        
        # Initialize trainer
        trainer = TinyLlamaTrainer(args.config)
        
        # Setup model and tokenizer
        trainer.setup_model_and_tokenizer(args.custom_tokenizer)
        
        # Prepare datasets
        train_dataset = trainer.prepare_dataset(args.train_data)
        eval_dataset = trainer.prepare_dataset(args.eval_data) if args.eval_data else None
        
        # Train model
        output_dir = trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=args.output_dir,
            run_name=args.run_name
        )
        
        print(f"Training completed successfully!")
        print(f"Model saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    with open("tinyllama_mobile/scripts/train_model.py", "w") as f:
        f.write(script)
    os.chmod("tinyllama_mobile/scripts/train_model.py", 0o755)
    print("Created scripts/train_model.py")

def create_mobile_converter():
    """Create mobile conversion script."""
    script = '''#!/usr/bin/env python3
"""
TinyLlama Mobile Converter
Converts trained TinyLlama models to mobile-optimized formats (CoreML, TensorFlow Lite).
"""

import argparse
import os
import sys
import torch
import numpy as np
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM

# Mobile deployment imports
try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False
    logging.warning("CoreMLTools not available - iOS deployment disabled")

try:
    import tensorflow as tf
    import onnx
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow/ONNX not available - Android deployment disabled")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TinyLlamaMobileConverter:
    """Converts TinyLlama models for mobile deployment."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def load_model(self, model_path: str):
        """Load the trained model and tokenizer."""
        logger.info(f"Loading model from {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # Use float32 for conversion
            device_map="cpu"  # Keep on CPU for conversion
        )
        self.model.eval()
        
    def convert_to_coreml(self, 
                         output_path: str,
                         max_length: int = 256,
                         quantize: bool = True) -> str:
        """Convert model to CoreML format for iOS."""
        if not COREML_AVAILABLE:
            raise ImportError("CoreMLTools not available")
        
        logger.info("Converting to CoreML...")
        
        # Create a simplified model for mobile deployment
        class MobileTinyLlama(torch.nn.Module):
            def __init__(self, model, tokenizer, max_length):
                super().__init__()
                self.model = model
                self.tokenizer = tokenizer
                self.max_length = max_length
                
            def forward(self, input_ids):
                # Simple forward pass for text generation
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids)
                    logits = outputs.logits
                    # Return next token probabilities
                    next_token_logits = logits[:, -1, :]
                    return torch.softmax(next_token_logits, dim=-1)
        
        # Create mobile-optimized model
        mobile_model = MobileTinyLlama(self.model, self.tokenizer, max_length)
        
        # Create example input
        example_input = torch.randint(0, self.tokenizer.vocab_size, (1, max_length))
        
        # Trace the model
        traced_model = torch.jit.trace(mobile_model, example_input)
        
        # Convert to CoreML
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=example_input.shape, dtype=np.int32)],
            minimum_deployment_target=ct.target.iOS15,
            compute_precision=ct.precision.FLOAT16 if quantize else ct.precision.FLOAT32
        )
        
        # Set metadata
        coreml_model.author = "TinyLlama Mobile"
        coreml_model.short_description = "TinyLlama model optimized for mobile devices"
        coreml_model.version = "1.0"
        
        # Apply quantization if requested
        if quantize and self.config['mobile']['ios']['quantization']['enabled']:
            bits = self.config['mobile']['ios']['quantization']['bits']
            coreml_model = ct.models.neural_network.quantization_utils.quantize_weights(
                coreml_model, nbits=bits
            )
        
        # Save model
        output_file = os.path.join(output_path, "TinyLlamaMobile.mlmodel")
        os.makedirs(output_path, exist_ok=True)
        coreml_model.save(output_file)
        
        logger.info(f"CoreML model saved to: {output_file}")
        
        # Save tokenizer info
        tokenizer_info = {
            "vocab_size": len(self.tokenizer),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "max_length": max_length
        }
        
        with open(os.path.join(output_path, "tokenizer_info.json"), "w") as f:
            json.dump(tokenizer_info, f, indent=2)
        
        return output_file
        
    def convert_to_tflite(self, 
                         output_path: str,
                         max_length: int = 256,
                         quantize: bool = True) -> str:
        """Convert model to TensorFlow Lite format for Android."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        logger.info("Converting to TensorFlow Lite...")
        
        # First convert to ONNX, then to TFLite
        onnx_path = self.convert_to_onnx(output_path, max_length)
        tflite_path = self.onnx_to_tflite(onnx_path, output_path, quantize)
        
        return tflite_path
        
    def convert_to_onnx(self, output_path: str, max_length: int = 256) -> str:
        """Convert to ONNX format."""
        logger.info("Converting to ONNX...")
        
        # Create simplified model for ONNX export
        class ONNXTinyLlama(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, input_ids):
                outputs = self.model(input_ids=input_ids)
                return outputs.logits
        
        onnx_model = ONNXTinyLlama(self.model)
        
        # Create example input
        example_input = torch.randint(0, self.tokenizer.vocab_size, (1, max_length))
        
        # Export to ONNX
        onnx_path = os.path.join(output_path, "tinyllama_mobile.onnx")
        os.makedirs(output_path, exist_ok=True)
        
        torch.onnx.export(
            onnx_model,
            example_input,
            onnx_path,
            export_params=True,
            opset_version=12,
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size', 1: 'sequence_length'}
            }
        )
        
        logger.info(f"ONNX model saved to: {onnx_path}")
        return onnx_path
        
    def onnx_to_tflite(self, onnx_path: str, output_path: str, quantize: bool = True) -> str:
        """Convert ONNX to TensorFlow Lite."""
        try:
            import onnx_tf
            
            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            
            # Convert to TensorFlow
            tf_rep = onnx_tf.backend.prepare(onnx_model)
            tf_model_path = os.path.join(output_path, "tf_model")
            tf_rep.export_graph(tf_model_path)
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
            
            # Apply optimizations
            if quantize and self.config['mobile']['android']['quantization']['enabled']:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                
                # Representative dataset for quantization
                def representative_dataset():
                    for _ in range(self.config['mobile']['android']['quantization']['representative_dataset_size']):
                        data = np.random.randint(
                            0, self.tokenizer.vocab_size, 
                            size=(1, 256), 
                            dtype=np.int32
                        )
                        yield [data.astype(np.float32)]
                
                converter.representative_dataset = representative_dataset
            
            # Convert
            tflite_model = converter.convert()
            
            # Save TFLite model
            tflite_path = os.path.join(output_path, "tinyllama_mobile.tflite")
            with open(tflite_path, "wb") as f:
                f.write(tflite_model)
            
            logger.info(f"TFLite model saved to: {tflite_path}")
            return tflite_path
            
        except ImportError:
            logger.error("onnx-tf not available. Install with: pip install onnx-tf")
            raise
            
    def optimize_for_mobile(self, model_path: str, output_path: str):
        """Apply mobile-specific optimizations."""
        logger.info("Applying mobile optimizations...")
        
        # Load model
        self.load_model(model_path)
        
        # Apply pruning if enabled
        if self.config['optimization']['pruning']['enabled']:
            self.apply_pruning()
        
        # Apply distillation if enabled
        if self.config['optimization']['distillation']['enabled']:
            self.apply_distillation()
            
        # Save optimized model
        optimized_path = os.path.join(output_path, "optimized")
        os.makedirs(optimized_path, exist_ok=True)
        
        self.model.save_pretrained(optimized_path)
        self.tokenizer.save_pretrained(optimized_path)
        
        logger.info(f"Optimized model saved to: {optimized_path}")
        return optimized_path
        
    def apply_pruning(self):
        """Apply structured pruning to reduce model size."""
        # Implement pruning logic here
        logger.info("Applying model pruning...")
        
    def apply_distillation(self):
        """Apply knowledge distillation."""
        # Implement distillation logic here
        logger.info("Applying knowledge distillation...")

def main():
    parser = argparse.ArgumentParser(description="Convert TinyLlama for mobile deployment")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--output_dir", type=str, default="models/mobile",
                       help="Output directory for mobile models")
    parser.add_argument("--platform", type=str, nargs="+", 
                       choices=["ios", "android", "both"],
                       default=["both"], help="Target platforms")
    parser.add_argument("--max_length", type=int, default=256,
                       help="Maximum sequence length for mobile models")
    parser.add_argument("--quantize", action="store_true", default=True,
                       help="Apply quantization")
    parser.add_argument("--optimize", action="store_true",
                       help="Apply mobile optimizations")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="Configuration file path")
    
    args = parser.parse_args()
    
    try:
        # Initialize converter
        converter = TinyLlamaMobileConverter(args.config)
        
        # Load model
        converter.load_model(args.model_path)
        
        # Apply optimizations if requested
        if args.optimize:
            model_path = converter.optimize_for_mobile(args.model_path, args.output_dir)
            converter.load_model(model_path)
        
        # Expand platform list
        platforms = []
        for platform in args.platform:
            if platform == "both":
                platforms.extend(["ios", "android"])
            else:
                platforms.append(platform)
        
        # Convert for each platform
        for platform in set(platforms):
            platform_dir = os.path.join(args.output_dir, platform)
            os.makedirs(platform_dir, exist_ok=True)
            
            if platform == "ios" and COREML_AVAILABLE:
                converter.convert_to_coreml(
                    platform_dir, 
                    args.max_length, 
                    args.quantize
                )
            elif platform == "android" and TF_AVAILABLE:
                converter.convert_to_tflite(
                    platform_dir,
                    args.max_length,
                    args.quantize
                )
            else:
                logger.warning(f"Skipping {platform} - required dependencies not available")
        
        print("Mobile conversion completed successfully!")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    with open("tinyllama_mobile/scripts/convert_to_mobile.py", "w") as f:
        f.write(script)
    os.chmod("tinyllama_mobile/scripts/convert_to_mobile.py", 0o755)
    print("Created scripts/convert_to_mobile.py")

def create_deployment_scripts():
    """Create iOS and Android deployment scripts."""
    
    # iOS deployment script
    ios_script = '''#!/usr/bin/env python3
"""
iOS Deployment Script for TinyLlama
Creates deployment package with CoreML model and Swift integration code.
"""

import os
import shutil
from pathlib import Path

def create_ios_deployment_package(model_dir: str, output_dir: str = "deployment/ios"):
    """Create iOS deployment package."""
    
    print("Creating iOS deployment package...")
    
    # Create directory structure
    ios_dir = Path(output_dir)
    (ios_dir / "Models").mkdir(parents=True, exist_ok=True)  
    (ios_dir / "Sources" / "TinyLlama").mkdir(parents=True, exist_ok=True)
    (ios_dir / "Examples").mkdir(parents=True, exist_ok=True)
    
    # Copy CoreML model
    model_files = list(Path(model_dir).glob("*.mlmodel"))
    if model_files:
        shutil.copy2(model_files[0], ios_dir / "Models")
        print(f"Copied CoreML model: {model_files[0].name}")
    
    # Copy tokenizer info
    tokenizer_file = Path(model_dir) / "tokenizer_info.json"
    if tokenizer_file.exists():
        shutil.copy2(tokenizer_file, ios_dir / "Models")
    
    # Create Swift Package manifest
    package_swift = f"""// swift-tools-version:5.7
import PackageDescription

let package = Package(
    name: "TinyLlama",
    platforms: [
        .iOS(.v15)
    ],
    products: [
        .library(
            name: "TinyLlama",
            targets: ["TinyLlama"]
        ),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "TinyLlama",
            dependencies: [],
            resources: [
                .process("Models")
            ]
        ),
    ]
)
"""
    
    with open(ios_dir / "Package.swift", "w") as f:
        f.write(package_swift)
    
    # Create Swift implementation
    swift_code = '''import CoreML
import Foundation

@available(iOS 15.0, *)
public class TinyLlama {
    private var model: MLModel?
    private let tokenizer: TinyLlamaTokenizer
    
    public init() throws {
        // Load CoreML model
        guard let modelURL = Bundle.module.url(forResource: "TinyLlamaMobile", withExtension: "mlmodel") else {
            throw TinyLlamaError.modelNotFound
        }
        
        let config = MLModelConfiguration()
        config.computeUnits = .all
        
        self.model = try MLModel(contentsOf: modelURL, configuration: config)
        self.tokenizer = try TinyLlamaTokenizer()
    }
    
    public func generate(prompt: String, maxTokens: Int = 50) async throws -> String {
        guard let model = model else {
            throw TinyLlamaError.modelNotLoaded
        }
        
        // Tokenize input
        var tokens = tokenizer.encode(prompt)
        
        // Generate tokens one by one
        for _ in 0..<maxTokens {
            // Prepare input
            let inputArray = try MLMultiArray(shape: [1, NSNumber(value: tokens.count)], dataType: .int32)
            for (i, token) in tokens.enumerated() {
                inputArray[i] = NSNumber(value: token)
            }
            
            // Get prediction
            let input = TinyLlamaMobileInput(input_ids: inputArray)
            let output = try model.prediction(from: input)
            
            // Get next token (simple greedy sampling)
            let probabilities = output.featureValue(for: "logits")?.multiArrayValue
            let nextToken = getNextToken(from: probabilities)
            
            tokens.append(nextToken)
            
            // Stop if EOS token
            if nextToken == tokenizer.eosTokenId {
                break
            }
        }
        
        return tokenizer.decode(tokens)
    }
    
    private func getNextToken(from probabilities: MLMultiArray?) -> Int {
        // Simple greedy decoding - return token with highest probability
        guard let probs = probabilities else { return 0 }
        
        var maxProb: Float = -Float.infinity
        var maxIndex: Int = 0
        
        for i in 0..<probs.count {
            let prob = probs[i].floatValue
            if prob > maxProb {
                maxProb = prob
                maxIndex = i
            }
        }
        
        return maxIndex
    }
}

// Simple tokenizer implementation
public class TinyLlamaTokenizer {
    private let vocabulary: [String: Int]
    private let reverseVocabulary: [Int: String]
    public let eosTokenId: Int
    
    init() throws {
        // Load tokenizer info
        guard let tokenizerURL = Bundle.module.url(forResource: "tokenizer_info", withExtension: "json") else {
            throw TinyLlamaError.tokenizerNotFound
        }
        
        let data = try Data(contentsOf: tokenizerURL)
        let tokenizerInfo = try JSONDecoder().decode(TokenizerInfo.self, from: data)
        
        // Initialize with basic vocabulary (simplified)
        var vocab: [String: Int] = [:]
        var reverseVocab: [Int: String] = [:]
        
        // Add basic tokens (this would normally be loaded from a proper vocab file)
        for i in 0..<tokenizerInfo.vocabSize {
            let token = "token_\\(i)"
            vocab[token] = i
            reverseVocab[i] = token
        }
        
        self.vocabulary = vocab
        self.reverseVocabulary = reverseVocab
        self.eosTokenId = tokenizerInfo.eosTokenId ?? 2
    }
    
    public func encode(_ text: String) -> [Int] {
        // Simple whitespace tokenization (would use SentencePiece in practice)
        let tokens = text.components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
        
        return tokens.compactMap { vocabulary[$0] ?? vocabulary["<unk>"] ?? 1 }
    }
    
    public func decode(_ tokens: [Int]) -> String {
        let words = tokens.compactMap { reverseVocabulary[$0] }
        return words.joined(separator: " ")
    }
}

public enum TinyLlamaError: Error {
    case modelNotFound
    case modelNotLoaded
    case tokenizerNotFound
}

private struct TokenizerInfo: Codable {
    let vocabSize: Int
    let padTokenId: Int?
    let eosTokenId: Int?
    let bosTokenId: Int?
    let maxLength: Int
}
'''
    
    with open(ios_dir / "Sources" / "TinyLlama" / "TinyLlama.swift", "w") as f:
        f.write(swift_code)
        
    # Create example usage
    example_code = '''import SwiftUI
import TinyLlama

struct ContentView: View {
    @State private var prompt = ""
    @State private var generatedText = ""
    @State private var isGenerating = false
    
    private let tinyLlama = try! TinyLlama()
    
    var body: some View {
        VStack(spacing: 20) {
            TextField("Enter your prompt", text: $prompt, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(3)
            
            Button("Generate") {
                generateText()
            }
            .disabled(prompt.isEmpty || isGenerating)
            
            if isGenerating {
                ProgressView("Generating...")
            }
            
            ScrollView {
                Text(generatedText)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(8)
            }
            
            Spacer()
        }
        .padding()
    }
    
    private func generateText() {
        isGenerating = true
        
        Task {
            do {
                let result = try await tinyLlama.generate(prompt: prompt, maxTokens: 50)
                await MainActor.run {
                    generatedText = result
                    isGenerating = false
                }
            } catch {
                await MainActor.run {
                    generatedText = "Error: \\(error.localizedDescription)"
                    isGenerating = false
                }
            }
        }
    }
}
'''
    
    with open(ios_dir / "Examples" / "ExampleApp.swift", "w") as f:
        f.write(example_code)
    
    print(f"iOS deployment package created at: {ios_dir}")
    
    return str(ios_dir)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python deploy_ios.py <model_directory>")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    create_ios_deployment_package(model_dir)
'''
    
    with open("tinyllama_mobile/scripts/deploy_ios.py", "w") as f:
        f.write(ios_script)
    os.chmod("tinyllama_mobile/scripts/deploy_ios.py", 0o755)
    
    # Android deployment script
    android_script = '''#!/usr/bin/env python3
"""
Android Deployment Script for TinyLlama
Creates deployment package with TensorFlow Lite model and Kotlin integration code.
"""

import os
import shutil
from pathlib import Path

def create_android_deployment_package(model_dir: str, output_dir: str = "deployment/android"):
    """Create Android deployment package."""
    
    print("Creating Android deployment package...")
    
    # Create directory structure
    android_dir = Path(output_dir)
    (android_dir / "src" / "main" / "assets").mkdir(parents=True, exist_ok=True)
    (android_dir / "src" / "main" / "java" / "com" / "tinyllama").mkdir(parents=True, exist_ok=True)
    (android_dir / "examples").mkdir(parents=True, exist_ok=True)
    
    # Copy TFLite model
    model_files = list(Path(model_dir).glob("*.tflite"))
    if model_files:
        shutil.copy2(model_files[0], android_dir / "src" / "main" / "assets")
        print(f"Copied TFLite model: {model_files[0].name}")
    
    # Copy tokenizer info
    tokenizer_file = Path(model_dir) / "tokenizer_info.json"
    if tokenizer_file.exists():
        shutil.copy2(tokenizer_file, android_dir / "src" / "main" / "assets")
    
    # Create build.gradle
    build_gradle = '''plugins {
    id 'com.android.library'
    id 'kotlin-android'
}

android {
    compileSdk 34

    defaultConfig {
        minSdk 24
        targetSdk 34
        
        testInstrumentationRunner "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            minifyEnabled false
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
        }
    }
    
    compileOptions {
        sourceCompatibility JavaVersion.VERSION_1_8
        targetCompatibility JavaVersion.VERSION_1_8
    }
    
    kotlinOptions {
        jvmTarget = '1.8'
    }
}

dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
    implementation 'androidx.core:core-ktx:1.12.0'
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3'
    implementation 'com.google.code.gson:gson:2.10.1'
    
    testImplementation 'junit:junit:4.13.2'
    androidTestImplementation 'androidx.test.ext:junit:1.1.5'
}
'''
    
    with open(android_dir / "build.gradle", "w") as f:
        f.write(build_gradle)
    
    # Create Kotlin implementation
    kotlin_code = '''package com.tinyllama

import android.content.Context
import android.content.res.AssetManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import com.google.gson.Gson
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.random.Random

class TinyLlama(private val context: Context) {
    
    private var interpreter: Interpreter? = null
    private var tokenizer: TinyLlamaTokenizer? = null
    private val maxLength = 256
    
    init {
        loadModel()
        loadTokenizer()
    }
    
    private fun loadModel() {
        try {
            val modelBuffer = loadModelFile("tinyllama_mobile.tflite")
            val options = Interpreter.Options().apply {
                // Use GPU delegate if available
                try {
                    addDelegate(org.tensorflow.lite.gpu.GpuDelegate())
                } catch (e: Exception) {
                    // Fallback to CPU
                }
            }
            
            interpreter = Interpreter(modelBuffer, options)
            println("TinyLlama model loaded successfully")
        } catch (e: Exception) {
            println("Error loading model: ${e.message}")
        }
    }
    
    private fun loadTokenizer() {
        try {
            val tokenizerJson = context.assets.open("tokenizer_info.json").bufferedReader().use { it.readText() }
            val tokenizerInfo = Gson().fromJson(tokenizerJson, TokenizerInfo::class.java)
            tokenizer = TinyLlamaTokenizer(tokenizerInfo)
            println("Tokenizer loaded successfully")
        } catch (e: Exception) {
            println("Error loading tokenizer: ${e.message}")
        }
    }
    
    private fun loadModelFile(fileName: String): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(fileName)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    suspend fun generate(prompt: String, maxTokens: Int = 50): String = withContext(Dispatchers.Default) {
        val interpreter = this@TinyLlama.interpreter ?: return@withContext "Model not loaded"
        val tokenizer = this@TinyLlama.tokenizer ?: return@withContext "Tokenizer not loaded"
        
        try {
            // Tokenize input
            val tokens = tokenizer.encode(prompt).toMutableList()
            
            // Generate tokens
            repeat(maxTokens) {
                if (tokens.size >= maxLength) return@repeat
                
                // Prepare input
                val inputArray = Array(1) { IntArray(maxLength) { 0 } }
                tokens.forEachIndexed { index, token ->
                    if (index < maxLength) {
                        inputArray[0][index] = token
                    }
                }
                
                // Run inference
                val outputArray = Array(1) { Array(maxLength) { FloatArray(tokenizer.vocabSize) { 0f } } }
                interpreter.run(inputArray, outputArray)
                
                // Get next token (simple greedy sampling)
                val lastTokenLogits = outputArray[0][tokens.size - 1]
                val nextToken = getNextToken(lastTokenLogits)
                
                tokens.add(nextToken)
                
                // Stop if EOS token
                if (nextToken == tokenizer.eosTokenId) {
                    break
                }
            }
            
            return@withContext tokenizer.decode(tokens)
            
        } catch (e: Exception) {
            return@withContext "Error during generation: ${e.message}"
        }
    }
    
    private fun getNextToken(logits: FloatArray): Int {
        // Simple greedy decoding
        var maxIndex = 0
        var maxValue = Float.NEGATIVE_INFINITY
        
        logits.forEachIndexed { index, value ->
            if (value > maxValue) {
                maxValue = value
                maxIndex = index
            }
        }
        
        return maxIndex
    }
    
    fun close() {
        interpreter?.close()
        interpreter = null
    }
}

class TinyLlamaTokenizer(private val tokenizerInfo: TokenizerInfo) {
    val vocabSize: Int = tokenizerInfo.vocabSize
    val eosTokenId: Int = tokenizerInfo.eosTokenId ?: 2
    private val bosTokenId: Int = tokenizerInfo.bosTokenId ?: 1
    private val padTokenId: Int = tokenizerInfo.padTokenId ?: 0
    
    fun encode(text: String): List<Int> {
        // Simple whitespace tokenization (would use SentencePiece in practice)
        val tokens = text.split(Regex("\\\\s+")).filter { it.isNotEmpty() }
        
        // Convert to IDs (simplified - would use actual vocabulary mapping)
        return tokens.map { token ->
            // Simple hash-based mapping for demonstration
            Math.abs(token.hashCode()) % (vocabSize - 10) + 10
        }
    }
    
    fun decode(tokens: List<Int>): String {
        // Simple reverse mapping (would use actual vocabulary in practice)
        return tokens.joinToString(" ") { "token_$it" }
    }
}

data class TokenizerInfo(
    val vocabSize: Int,
    val padTokenId: Int?,
    val eosTokenId: Int?,
    val bosTokenId: Int?,
    val maxLength: Int
)
'''
    
    with open(android_dir / "src" / "main" / "java" / "com" / "tinyllama" / "TinyLlama.kt", "w") as f:
        f.write(kotlin_code)
    
    # Create example usage
    example_code = '''package com.tinyllama.example

import android.os.Bundle
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.tinyllama.TinyLlama
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {
    
    private lateinit var tinyLlama: TinyLlama
    private lateinit var promptEditText: EditText
    private lateinit var generateButton: Button
    private lateinit var resultTextView: TextView
    private lateinit var progressBar: ProgressBar
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        initViews()
        initTinyLlama()
        setupClickListeners()
    }
    
    private fun initViews() {
        promptEditText = findViewById(R.id.promptEditText)
        generateButton = findViewById(R.id.generateButton)
        resultTextView = findViewById(R.id.resultTextView)
        progressBar = findViewById(R.id.progressBar)
    }
    
    private fun initTinyLlama() {
        tinyLlama = TinyLlama(this)
    }
    
    private fun setupClickListeners() {
        generateButton.setOnClickListener {
            val prompt = promptEditText.text.toString().trim()
            if (prompt.isNotEmpty()) {
                generateText(prompt)
            } else {
                Toast.makeText(this, "Please enter a prompt", Toast.LENGTH_SHORT).show()
            }
        }
    }
    
    private fun generateText(prompt: String) {
        generateButton.isEnabled = false
        progressBar.visibility = ProgressBar.VISIBLE
        resultTextView.text = "Generating..."
        
        lifecycleScope.launch {
            try {
                val result = tinyLlama.generate(prompt, maxTokens = 50)
                resultTextView.text = result
            } catch (e: Exception) {
                resultTextView.text = "Error: ${e.message}"
            } finally {
                generateButton.isEnabled = true
                progressBar.visibility = ProgressBar.GONE
            }
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        tinyLlama.close()
    }
}
'''
    
    with open(android_dir / "examples" / "MainActivity.kt", "w") as f:
        f.write(example_code)
    
    # Create layout file
    layout_xml = '''<?xml version="1.0" encoding="utf-8"?>
<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical"
    android:padding="16dp">

    <TextView
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="TinyLlama Mobile Demo"
        android:textSize="24sp"
        android:textStyle="bold"
        android:layout_marginBottom="24dp" />

    <EditText
        android:id="@+id/promptEditText"
        android:layout_width="
