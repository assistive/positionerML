# bert_mobile/src/tokenizer_utils.py

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from collections import defaultdict

import numpy as np
from transformers import BertTokenizer, BertTokenizerFast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BERTTokenizerUtils:
    """Utilities for BERT tokenizer operations and custom vocabulary handling."""
    
    def __init__(self, tokenizer_path: str):
        """
        Initialize tokenizer utilities.
        
        Args:
            tokenizer_path: Path to tokenizer directory or model
        """
        self.tokenizer_path = tokenizer_path
        self.tokenizer = self.load_tokenizer()
        
    def load_tokenizer(self):
        """Load tokenizer from path."""
        try:
            if os.path.isfile(self.tokenizer_path) and self.tokenizer_path.endswith('.txt'):
                # Load from vocab file
                return BertTokenizer.from_pretrained(
                    'bert-base-uncased',  # Base tokenizer
                    vocab_file=self.tokenizer_path,
                    do_lower_case=True
                )
            else:
                # Load from directory
                return BertTokenizer.from_pretrained(self.tokenizer_path)
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise
    
    def analyze_tokenization(self, texts: List[str]) -> Dict:
        """Analyze tokenization statistics for given texts."""
        logger.info("Analyzing tokenization...")
        
        stats = {
            'total_texts': len(texts),
            'total_tokens': 0,
            'total_subwords': 0,
            'avg_tokens_per_text': 0,
            'max_tokens': 0,
            'min_tokens': float('inf'),
            'oov_count': 0,
            'subword_ratio': 0,
            'length_distribution': defaultdict(int)
        }
        
        all_tokens = []
        
        for text in texts:
            tokens = self.tokenizer.tokenize(text)
            token_count = len(tokens)
            
            all_tokens.extend(tokens)
            stats['total_tokens'] += token_count
            stats['max_tokens'] = max(stats['max_tokens'], token_count)
            stats['min_tokens'] = min(stats['min_tokens'], token_count)
            
            # Count subword tokens (starting with ##)
            subword_count = sum(1 for token in tokens if token.startswith('##'))
            stats['total_subwords'] += subword_count
            
            # Count OOV tokens
            oov_count = sum(1 for token in tokens if token == '[UNK]')
            stats['oov_count'] += oov_count
            
            # Length distribution
            length_bucket = (token_count // 10) * 10  # Group by 10s
            stats['length_distribution'][length_bucket] += 1
        
        # Calculate averages and ratios
        if stats['total_texts'] > 0:
            stats['avg_tokens_per_text'] = stats['total_tokens'] / stats['total_texts']
            stats['subword_ratio'] = stats['total_subwords'] / stats['total_tokens']
            stats['oov_ratio'] = stats['oov_count'] / stats['total_tokens']
        
        # Most common tokens
        from collections import Counter
        token_counts = Counter(all_tokens)
        stats['most_common_tokens'] = token_counts.most_common(20)
        
        return stats
    
    def validate_custom_vocabulary(self, vocab_file: str, test_texts: List[str]) -> Dict:
        """Validate custom vocabulary against test texts."""
        logger.info("Validating custom vocabulary...")
        
        # Load custom tokenizer
        custom_tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased',
            vocab_file=vocab_file,
            do_lower_case=True
        )
        
        # Compare with base tokenizer
        base_stats = self.analyze_tokenization(test_texts)
        
        # Temporarily switch to custom tokenizer
        original_tokenizer = self.tokenizer
        self.tokenizer = custom_tokenizer
        custom_stats = self.analyze_tokenization(test_texts)
        self.tokenizer = original_tokenizer
        
        validation_results = {
            'base_tokenizer': base_stats,
            'custom_tokenizer': custom_stats,
            'improvements': {
                'oov_reduction': base_stats['oov_ratio'] - custom_stats['oov_ratio'],
                'avg_length_change': custom_stats['avg_tokens_per_text'] - base_stats['avg_tokens_per_text'],
                'subword_ratio_change': custom_stats['subword_ratio'] - base_stats['subword_ratio']
            }
        }
        
        return validation_results
    
    def create_mobile_tokenizer_config(self, output_dir: str):
        """Create tokenizer configuration optimized for mobile deployment."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create simplified tokenizer config
        mobile_config = {
            'vocab_size': len(self.tokenizer.vocab),
            'do_lower_case': getattr(self.tokenizer, 'do_lower_case', True),
            'special_tokens': {
                'pad_token': self.tokenizer.pad_token,
                'unk_token': self.tokenizer.unk_token,
                'cls_token': self.tokenizer.cls_token,
                'sep_token': self.tokenizer.sep_token,
                'mask_token': getattr(self.tokenizer, 'mask_token', '[MASK]'),
                'pad_token_id': self.tokenizer.pad_token_id,
                'unk_token_id': self.tokenizer.unk_token_id,
                'cls_token_id': self.tokenizer.cls_token_id,
                'sep_token_id': self.tokenizer.sep_token_id,
                'mask_token_id': getattr(self.tokenizer, 'mask_token_id', 103)
            },
            'max_length': getattr(self.tokenizer, 'model_max_length', 512)
        }
        
        # Save config
        config_path = output_path / "tokenizer_config.json"
        with open(config_path, 'w') as f:
            json.dump(mobile_config, f, indent=2)
        
        # Save vocabulary as simple text file for mobile
        vocab = self.tokenizer.get_vocab()
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        
        vocab_path = output_path / "vocab.txt"
        with open(vocab_path, 'w', encoding='utf-8') as f:
            for token, _ in sorted_vocab:
                f.write(f"{token}\n")
        
        # Save vocabulary as JSON for easier parsing
        vocab_json_path = output_path / "vocab.json"
        with open(vocab_json_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Mobile tokenizer config saved to {output_path}")
        return str(output_path)
    
    def optimize_for_domain(self, domain_texts: List[str], 
                           output_vocab_path: str,
                           target_vocab_size: int = 30000) -> str:
        """Optimize tokenizer vocabulary for specific domain."""
        logger.info("Optimizing vocabulary for domain...")
        
        # Analyze domain-specific terms
        domain_stats = self.analyze_tokenization(domain_texts)
        
        # Extract frequent domain terms that are tokenized poorly
        poorly_tokenized = []
        
        for text in domain_texts[:1000]:  # Sample for analysis
            tokens = self.tokenizer.tokenize(text)
            words = text.lower().split()
            
            for word in words:
                word_tokens = self.tokenizer.tokenize(word)
                # If word is split into many subword tokens, it might benefit from being in vocab
                if len(word_tokens) > 2 and len(word) > 3:
                    poorly_tokenized.append(word)
        
        # Count frequency of poorly tokenized terms
        from collections import Counter
        term_counts = Counter(poorly_tokenized)
        
        # Get existing vocabulary
        existing_vocab = set(self.tokenizer.vocab.keys())
        
        # Add high-frequency domain terms not in vocabulary
        domain_terms = []
        for term, count in term_counts.most_common(1000):
            if term not in existing_vocab and count > 5:
                domain_terms.append(term)
        
        logger.info(f"Found {len(domain_terms)} domain-specific terms to add")
        
        # Create new vocabulary (simplified approach)
        # In practice, you'd retrain the tokenizer with domain data
        new_vocab = dict(self.tokenizer.vocab)
        
        # Add domain terms (replace least frequent tokens)
        vocab_items = sorted(new_vocab.items(), key=lambda x: x[1])
        
        # Remove some least important tokens to make room
        tokens_to_remove = len(domain_terms)
        for token, _ in vocab_items[-tokens_to_remove:]:
            if not token.startswith('[') and not token.startswith('##'):
                del new_vocab[token]
        
        # Add domain terms
        next_id = max(new_vocab.values()) + 1
        for term in domain_terms[:tokens_to_remove]:
            new_vocab[term] = next_id
            next_id += 1
        
        # Save optimized vocabulary
        output_path = Path(output_vocab_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        sorted_vocab = sorted(new_vocab.items(), key=lambda x: x[1])
        with open(output_path, 'w', encoding='utf-8') as f:
            for token, _ in sorted_vocab:
                f.write(f"{token}\n")
        
        logger.info(f"Optimized vocabulary saved to {output_path}")
        return str(output_path)
    
    def convert_to_sentencepiece(self, output_dir: str, texts: List[str]):
        """Convert to SentencePiece format for better mobile compatibility."""
        try:
            import sentencepiece as spm
        except ImportError:
            logger.error("SentencePiece not available. Install with: pip install sentencepiece")
            return None
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare training corpus
        corpus_file = output_path / "training_corpus.txt"
        with open(corpus_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        
        # Train SentencePiece model
        model_prefix = str(output_path / "sp_tokenizer")
        
        spm.SentencePieceTrainer.train(
            input=str(corpus_file),
            model_prefix=model_prefix,
            vocab_size=len(self.tokenizer.vocab),
            model_type='bpe',
            character_coverage=0.995,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            user_defined_symbols=['[CLS]', '[SEP]', '[MASK]']
        )
        
        logger.info(f"SentencePiece model saved to {model_prefix}")
        return model_prefix + ".model"

class BERTDataProcessor:
    """Process data for BERT training with custom tokenizers."""
    
    def __init__(self, tokenizer_path: str, max_length: int = 512):
        self.tokenizer_utils = BERTTokenizerUtils(tokenizer_path)
        self.tokenizer = self.tokenizer_utils.tokenizer
        self.max_length = max_length
    
    def process_texts_for_classification(self, texts: List[str], 
                                       labels: List[int]) -> List[Dict]:
        """Process texts and labels for classification task."""
        processed_data = []
        
        for text, label in zip(texts, labels):
            # Tokenize
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            processed_data.append({
                'text': text,
                'label': label,
                'input_ids': encoding['input_ids'].squeeze().tolist(),
                'attention_mask': encoding['attention_mask'].squeeze().tolist()
            })
        
        return processed_data
    
    def process_texts_for_mlm(self, texts: List[str], 
                             mlm_probability: float = 0.15) -> List[Dict]:
        """Process texts for masked language modeling."""
        processed_data = []
        
        for text in texts:
            # Tokenize
            tokens = self.tokenizer.tokenize(text)
            
            # Add special tokens
            tokens = ['[CLS]'] + tokens[:self.max_length-2] + ['[SEP]']
            
            # Create labels for MLM (only predict masked tokens)
            labels = [-100] * len(tokens)  # -100 = ignore in loss
            
            # Apply masking
            for i in range(len(tokens)):
                if tokens[i] in ['[CLS]', '[SEP]']:
                    continue
                
                if np.random.random() < mlm_probability:
                    labels[i] = self.tokenizer.vocab[tokens[i]]  # Original token
                    
                    # 80% mask, 10% random, 10% keep
                    rand = np.random.random()
                    if rand < 0.8:
                        tokens[i] = '[MASK]'
                    elif rand < 0.9:
                        # Random token
                        tokens[i] = self.tokenizer.convert_ids_to_tokens(
                            [np.random.randint(len(self.tokenizer.vocab))]
                        )[0]
            
            # Convert to IDs
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            
            # Pad
            while len(input_ids) < self.max_length:
                input_ids.append(0)
                labels.append(-100)
            
            attention_mask = [1 if id != 0 else 0 for id in input_ids]
            
            processed_data.append({
                'text': text,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            })
        
        return processed_data

# bert_mobile/src/model_utils.py

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from typing import Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class MobileBertConfig(BertConfig):
    """Configuration for mobile-optimized BERT."""
    
    def __init__(self, 
                 vocab_size: int = 30000,
                 hidden_size: int = 512,
                 num_hidden_layers: int = 6,
                 num_attention_heads: int = 8,
                 intermediate_size: int = 2048,
                 **kwargs):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            **kwargs
        )

class MobileBertModel(nn.Module):
    """Mobile-optimized BERT model."""
    
    def __init__(self, config: MobileBertConfig):
        super().__init__()
        self.config = config
        self.bert = BertModel(config)
        
        # Mobile optimizations
        self.apply_mobile_optimizations()
    
    def apply_mobile_optimizations(self):
        """Apply mobile-specific optimizations."""
        # Reduce precision of some operations
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Apply weight quantization awareness
                module.weight.data = module.weight.data.half().float()
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

class BertForMobileClassification(nn.Module):
    """BERT model for mobile classification tasks."""
    
    def __init__(self, config: MobileBertConfig, num_labels: int = 2):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        
        self.bert = MobileBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights."""
        self.classifier.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.classifier.bias.data.zero_()
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions
        }

def create_mobile_bert_model(vocab_size: int = 30000,
                           num_labels: int = 2,
                           hidden_size: int = 512,
                           num_layers: int = 6) -> BertForMobileClassification:
    """Create a mobile-optimized BERT model."""
    
    config = MobileBertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=hidden_size // 64,  # Ensure divisibility
        intermediate_size=hidden_size * 4
    )
    
    model = BertForMobileClassification(config, num_labels)
    
    logger.info(f"Created mobile BERT model:")
    logger.info(f"  Vocab size: {vocab_size}")
    logger.info(f"  Hidden size: {hidden_size}")
    logger.info(f"  Layers: {num_layers}")
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

def quantize_model(model: nn.Module) -> nn.Module:
    """Apply dynamic quantization to model."""
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )
    
    logger.info("Applied dynamic quantization to model")
    return quantized_model

def prune_model(model: nn.Module, sparsity: float = 0.2) -> nn.Module:
    """Apply structured pruning to model."""
    try:
        import torch.nn.utils.prune as prune
        
        # Apply pruning to linear layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=sparsity)
                prune.remove(module, 'weight')
        
        logger.info(f"Applied {sparsity*100}% pruning to model")
        
    except ImportError:
        logger.warning("Pruning not available in this PyTorch version")
    
    return model

# bert_mobile/notebooks/vocabulary_analysis.ipynb

{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# BERT Mobile Vocabulary Analysis\n",
    "\n",
    "This notebook analyzes vocabulary usage and provides insights for mobile optimization."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.append('../src')\n",
    "\n",
    "import json\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from pathlib import Path\n",
    "import numpy as np\n",
    "from collections import Counter\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "from tokenizer_utils import BERTTokenizerUtils\n",
    "from vocab_builder import VocabularyBuilder\n",
    "\n",
    "plt.style.use('seaborn-v0_8')\n",
    "sns.set_palette(\"husl\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Load and Analyze Base Vocabulary"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize tokenizer utils with base BERT\n",
    "tokenizer_utils = BERTTokenizerUtils('bert-base-uncased')\n",
    "\n",
    "# Basic tokenizer info\n",
    "print(f\"Vocabulary size: {len(tokenizer_utils.tokenizer.vocab):,}\")\n",
    "print(f\"Model max length: {tokenizer_utils.tokenizer.model_max_length}\")\n",
    "print(f\"Special tokens: {tokenizer_utils.tokenizer.special_tokens_map}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Analyze Tokenization on Sample Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Sample texts for analysis\n",
    "sample_texts = [\n",
    "    \"The quick brown fox jumps over the lazy dog.\",\n",
    "    \"Machine learning and artificial intelligence are transforming technology.\",\n",
    "    \"Natural language processing enables computers to understand human language.\",\n",
    "    \"BERT is a transformer-based model for natural language understanding.\",\n",
    "    \"Mobile deployment requires model optimization and quantization techniques.\",\n",
    "    \"TensorFlow Lite and CoreML provide mobile inference capabilities.\",\n",
    "    \"Vocabulary optimization can significantly improve model performance.\",\n",
    "    \"Subword tokenization helps handle out-of-vocabulary words effectively.\"\n",
    "]\n",
    "\n",
    "# Analyze tokenization\n",
    "stats = tokenizer_utils.analyze_tokenization(sample_texts)\n",
    "\n",
    "print(\"Tokenization Statistics:\")\n",
    "print(f\"Total texts: {stats['total_texts']}\")\n",
    "print(f\"Total tokens: {stats['total_tokens']}\")\n",
    "print(f\"Average tokens per text: {stats['avg_tokens_per_text']:.2f}\")\n",
    "print(f\"Subword ratio: {stats['subword_ratio']:.2%}\")\n",
    "print(f\"OOV ratio: {stats['oov_ratio']:.2%}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize token length distribution\n",
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))\n",
    "\n",
    "# Length distribution\n",
    "lengths = list(stats['length_distribution'].keys())\n",
    "counts = list(stats['length_distribution'].values())\n",
    "\n",
    "ax1.bar(lengths, counts)\n",
    "ax1.set_xlabel('Token Length (grouped by 10s)')\n",
    "ax1.set_ylabel('Number of Texts')\n",
    "ax1.set_title('Token Length Distribution')\n",
    "\n",
    "# Most common tokens\n",
    "tokens, token_counts = zip(*stats['most_common_tokens'][:15])\n",
    "ax2.barh(range(len(tokens)), token_counts)\n",
    "ax2.set_yticks(range(len(tokens)))\n",
    "ax2.set_yticklabels(tokens)\n",
    "ax2.set_xlabel('Frequency')\n",
    "ax2.set_title('Most Common Tokens')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Analyze Domain-Specific Vocabulary"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Domain-specific texts (AI/ML domain)\n",
    "domain_texts = [\n",
    "    \"Deep learning neural networks require GPU computation for training.\",\n",
    "    \"Convolutional neural networks excel at computer vision tasks.\",\n",
    "    \"Recurrent neural networks process sequential data effectively.\",\n",
    "    \"Transformer architectures revolutionized natural language processing.\",\n",
    "    \"Attention mechanisms help models focus on relevant information.\",\n",
    "    \"Pre-trained models can be fine-tuned for specific tasks.\",\n",
    "    \"Gradient descent optimizes model parameters during training.\",\n",
    "    \"Regularization techniques prevent overfitting in machine learning.\",\n",
    "    \"Cross-validation helps evaluate model generalization performance.\",\n",
    "    \"Hyperparameter tuning improves model accuracy and efficiency.\"\n",
    "]\n",
    "\n",
    "# Analyze domain tokenization\n",
    "domain_stats = tokenizer_utils.analyze_tokenization(domain_texts)\n",
    "\n",
    "print(\"Domain-Specific Tokenization:\")\n",
    "print(f\"Average tokens per text: {domain_stats['avg_tokens_per_text']:.2f}\")\n",
    "print(f\"Subword ratio: {domain_stats['subword_ratio']:.2%}\")\n",
    "print(f\"OOV ratio: {domain_stats['oov_ratio']:.2%}\")\n",
    "\n",
    "print(\"\\nMost common domain tokens:\")\n",
    "for token, count in domain_stats['most_common_tokens'][:10]:\n",
    "    print(f\"  {token}: {count}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Compare Base vs Domain Tokenization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Compare general vs domain-specific tokenization\n",
    "comparison_data = {\n",
    "    'Metric': ['Avg Tokens/Text', 'Subword Ratio', 'OOV Ratio'],\n",
    "    'General': [stats['avg_tokens_per_text'], stats['subword_ratio'], stats['oov_ratio']],\n",
    "    'Domain (AI/ML)': [domain_stats['avg_tokens_per_text'], domain_stats['subword_ratio'], domain_stats['oov_ratio']]\n",
    "}\n",
    "\n",
    "df_comparison = pd.DataFrame(comparison_data)\n",
    "print(\"Tokenization Comparison:\")\n",
    "print(df_comparison.to_string(index=False))\n",
    "\n",
    "# Visualize comparison\n",
    "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
    "\n",
    "for i, metric in enumerate(['Avg Tokens/Text', 'Subword Ratio', 'OOV Ratio']):\n",
    "    general_val = df_comparison[df_comparison['Metric'] == metric]['General'].values[0]\n",
    "    domain_val = df_comparison[df_comparison['Metric'] == metric]['Domain (AI/ML)'].values[0]\n",
    "    \n",
    "    axes[i].bar(['General', 'Domain'], [general_val, domain_val])\n",
    "    axes[i].set_title(metric)\n",
    "    axes[i].set_ylabel('Value')\n",
    "    \n",
    "    # Add value labels\n",
    "    axes[i].text(0, general_val + max(general_val, domain_val) * 0.02, f'{general_val:.3f}', ha='center')\n",
    "    axes[i].text(1, domain_val + max(general_val, domain_val) * 0.02, f'{domain_val:.3f}', ha='center')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Vocabulary Optimization Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Identify words that are poorly tokenized\n",
    "poorly_tokenized = []\n",
    "\n",
    "all_texts = sample_texts + domain_texts\n",
    "\n",
    "for text in all_texts:\n",
    "    words = text.lower().replace('.', '').replace(',', '').split()\n",
    "    for word in words:\n",
    "        tokens = tokenizer_utils.tokenizer.tokenize(word)\n",
    "        if len(tokens) > 2 and len(word) > 5:  # Word split into many pieces\n",
    "            poorly_tokenized.append((word, len(tokens), tokens))\n",
    "\n",
    "# Show poorly tokenized words\n",
    "print(\"Words that might benefit from being in vocabulary:\")\n",
    "for word, num_tokens, tokens in poorly_tokenized:\n",
    "    print(f\"  {word} -> {tokens} ({num_tokens} tokens)\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Mobile Optimization Recommendations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculate mobile optimization metrics\n",
    "current_vocab_size = len(tokenizer_utils.tokenizer.vocab)\n",
    "avg_sequence_length = (stats['avg_tokens_per_text'] + domain_stats['avg_tokens_per_text']) / 2\n",
    "\n",
    "print(\"=== MOBILE OPTIMIZATION ANALYSIS ===\")\n",
    "print(f\"Current vocabulary size: {current_vocab_size:,}\")\n",
    "print(f\"Average sequence length: {avg_sequence_length:.1f} tokens\")\n",
    "print(f\"Estimated model size: ~{current_vocab_size * 768 * 4 / (1024**2):.1f}MB (embeddings only)\")\n",
    "\n",
    "print(\"\\n=== RECOMMENDATIONS ===\")\n",
    "\n",
    "# Vocabulary size recommendations\n",
    "if current_vocab_size > 25000:\n",
    "    print(\"📝 Consider reducing vocabulary size to 15K-25K for mobile deployment\")\n",
    "    print(\"   - Use domain-specific vocabulary\")\n",
    "    print(\"   - Remove rare tokens\")\n",
    "    print(\"   - Use subword tokenization effectively\")\n",
    "\n",
    "# Sequence length recommendations  \n",
    "if avg_sequence_length > 100:\n",
    "    print(\"📏 Consider optimizing for shorter sequences (64-128 tokens)\")\n",
    "    print(\"   - Truncate long texts appropriately\")\n",
    "    print(\"   - Use sliding window for long documents\")\n",
    "\n",
    "# Subword ratio analysis\n",
    "if domain_stats['subword_ratio'] > 0.4:\n",
    "    print(\"🔤 High subword ratio detected\")\n",
    "    print(\"   - Add domain-specific terms to vocabulary\")\n",
    "    print
