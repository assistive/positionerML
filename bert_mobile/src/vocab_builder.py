# src/vocab_builder.py

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from collections import Counter, defaultdict
import logging
import re
import yaml

import pandas as pd
import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer, pre_tokenizers, processors, models, trainers
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
from transformers import BertTokenizer, BertTokenizerFast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VocabularyBuilder:
    """Build custom vocabulary for BERT models from text corpora."""
    
    def __init__(self, config_path: str = "config/vocab_config.yaml"):
        """
        Initialize vocabulary builder.
        
        Args:
            config_path: Path to vocabulary configuration file
        """
        self.config = self.load_config(config_path)
        self.vocab_size = self.config['vocabulary']['vocab_size']
        self.min_frequency = self.config['vocabulary']['min_frequency']
        self.special_tokens = self.config['vocabulary']['special_tokens']
        
    def load_config(self, config_path: str) -> Dict:
        """Load vocabulary configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def build_from_corpus(self, 
                         corpus_paths: List[str],
                         output_dir: str,
                         vocab_name: str = "custom_vocab") -> str:
        """
        Build vocabulary from text corpus.
        
        Args:
            corpus_paths: List of paths to text files
            output_dir: Output directory for vocabulary files
            vocab_name: Name for the vocabulary
            
        Returns:
            Path to the main vocabulary file
        """
        logger.info("Building vocabulary from corpus...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Collect all text data
        all_texts = self.load_texts(corpus_paths)
        
        # Preprocess texts
        processed_texts = self.preprocess_texts(all_texts)
        
        # Build WordPiece tokenizer
        tokenizer = self.build_wordpiece_tokenizer(processed_texts)
        
        # Save tokenizer and vocabulary
        vocab_path = self.save_vocabulary(tokenizer, output_path, vocab_name)
        
        # Generate vocabulary statistics
        self.generate_vocab_stats(tokenizer, processed_texts, output_path)
        
        logger.info(f"Vocabulary built successfully: {vocab_path}")
        return vocab_path
    
    def load_texts(self, corpus_paths: List[str]) -> List[str]:
        """Load texts from corpus files."""
        logger.info("Loading corpus texts...")
        
        all_texts = []
        
        for path in corpus_paths:
            path_obj = Path(path)
            
            if path_obj.is_file():
                all_texts.extend(self.load_file(path_obj))
            elif path_obj.is_dir():
                # Load all text files in directory
                for file_path in path_obj.rglob("*.txt"):
                    all_texts.extend(self.load_file(file_path))
                for file_path in path_obj.rglob("*.json"):
                    all_texts.extend(self.load_json_file(file_path))
                for file_path in path_obj.rglob("*.csv"):
                    all_texts.extend(self.load_csv_file(file_path))
        
        logger.info(f"Loaded {len(all_texts)} text samples")
        return all_texts
    
    def load_file(self, file_path: Path) -> List[str]:
        """Load text from a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix == '.txt':
                    # Split by lines and filter empty lines
                    lines = [line.strip() for line in f.readlines()]
                    return [line for line in lines if line]
                else:
                    # Read entire file as one text
                    return [f.read().strip()]
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
            return []
    
    def load_json_file(self, file_path: Path) -> List[str]:
        """Load text from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            texts = []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        texts.append(item)
                    elif isinstance(item, dict):
                        # Extract text from common keys
                        for key in ['text', 'content', 'body', 'message']:
                            if key in item:
                                texts.append(str(item[key]))
                                break
            elif isinstance(data, dict):
                # Extract text from common keys
                for key in ['text', 'content', 'body', 'message']:
                    if key in data:
                        if isinstance(data[key], list):
                            texts.extend([str(t) for t in data[key]])
                        else:
                            texts.append(str(data[key]))
            
            return texts
        except Exception as e:
            logger.warning(f"Error loading JSON {file_path}: {e}")
            return []
    
    def load_csv_file(self, file_path: Path) -> List[str]:
        """Load text from CSV file."""
        try:
            df = pd.read_csv(file_path)
            
            # Look for text columns
            text_columns = []
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['text', 'content', 'body', 'message', 'comment']):
                    text_columns.append(col)
            
            if not text_columns:
                # Use all string columns
                text_columns = [col for col in df.columns if df[col].dtype == 'object']
            
            texts = []
            for col in text_columns:
                texts.extend(df[col].dropna().astype(str).tolist())
            
            return texts
        except Exception as e:
            logger.warning(f"Error loading CSV {file_path}: {e}")
            return []
    
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """Preprocess texts according to configuration."""
        logger.info("Preprocessing texts...")
        
        preprocessing_config = self.config['vocabulary']['preprocessing']
        processed_texts = []
        
        for text in tqdm(texts, desc="Preprocessing"):
            # Remove URLs
            if preprocessing_config.get('remove_urls', True):
                text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            
            # Remove emails
            if preprocessing_config.get('remove_emails', True):
                text = re.sub(r'\S+@\S+', '', text)
            
            # Remove phone numbers
            if preprocessing_config.get('remove_phone_numbers', True):
                text = re.sub(r'[\+]?[1-9]?[0-9]{7,15}', '', text)
            
            # Remove special characters
            if preprocessing_config.get('remove_special_chars', False):
                text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\;\:]', '', text)
            
            # Normalize whitespace
            if preprocessing_config.get('normalize_whitespace', True):
                text = re.sub(r'\s+', ' ', text)
            
            # Strip and filter empty
            text = text.strip()
            if text:
                processed_texts.append(text)
        
        logger.info(f"Preprocessed {len(processed_texts)} texts")
        return processed_texts
    
    def build_wordpiece_tokenizer(self, texts: List[str]) -> Tokenizer:
        """Build WordPiece tokenizer from texts."""
        logger.info("Building WordPiece tokenizer...")
        
        # Initialize tokenizer
        tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        
        # Set up normalization
        normalizers = []
        if self.config['vocabulary']['tokenization']['strip_accents']:
            normalizers.extend([NFD(), StripAccents()])
        if self.config['vocabulary']['tokenization']['do_lower_case']:
            normalizers.append(Lowercase())
        
        if normalizers:
            tokenizer.normalizer = Sequence(normalizers)
        
        # Set up pre-tokenization
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        
        # Set up post-processing
        tokenizer.post_processor = processors.BertProcessing(
            sep=("[SEP]", self.special_tokens.index("[SEP]")),
            cls=("[CLS]", self.special_tokens.index("[CLS]"))
        )
        
        # Configure trainer
        trainer = trainers.WordPieceTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens + self.config['vocabulary'].get('domain_tokens', []),
            continuing_subword_prefix="##",
            end_of_word_suffix=None
        )
        
        # Train tokenizer
        tokenizer.train_from_iterator(texts, trainer=trainer)
        
        logger.info(f"Built tokenizer with {tokenizer.get_vocab_size()} tokens")
        return tokenizer
    
    def save_vocabulary(self, 
                       tokenizer: Tokenizer, 
                       output_dir: Path, 
                       vocab_name: str) -> str:
        """Save vocabulary and tokenizer files."""
        logger.info("Saving vocabulary files...")
        
        # Save main vocabulary file (compatible with BERT)
        vocab_path = output_dir / f"{vocab_name}.txt"
        vocab = tokenizer.get_vocab()
        
        # Sort by token ID
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        
        with open(vocab_path, 'w', encoding='utf-8') as f:
            for token, _ in sorted_vocab:
                f.write(f"{token}\n")
        
        # Save tokenizer JSON (for fast loading)
        if self.config['vocabulary']['output']['save_tokenizer_json']:
            tokenizer_path = output_dir / f"{vocab_name}_tokenizer.json"
            tokenizer.save(str(tokenizer_path))
        
        # Save as Transformers tokenizer
        transformers_tokenizer = BertTokenizerFast(tokenizer_object=tokenizer)
        transformers_dir = output_dir / f"{vocab_name}_transformers"
        transformers_tokenizer.save_pretrained(str(transformers_dir))
        
        # Save vocabulary mapping
        vocab_mapping_path = output_dir / f"{vocab_name}_mapping.json"
        with open(vocab_mapping_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        
        # Save configuration used
        config_path = output_dir / f"{vocab_name}_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"Vocabulary saved to {vocab_path}")
        return str(vocab_path)
    
    def generate_vocab_stats(self, 
                           tokenizer: Tokenizer, 
                           texts: List[str], 
                           output_dir: Path):
        """Generate vocabulary statistics."""
        if not self.config['vocabulary']['output']['save_statistics']:
            return
        
        logger.info("Generating vocabulary statistics...")
        
        vocab = tokenizer.get_vocab()
        stats = {
            'vocab_size': len(vocab),
            'special_tokens': self.special_tokens,
            'num_texts': len(texts),
            'config': self.config
        }
        
        # Token frequency analysis
        token_counts = Counter()
        total_tokens = 0
        
        for text in tqdm(texts[:1000], desc="Analyzing tokens"):  # Sample for speed
            encoding = tokenizer.encode(text)
            tokens = encoding.tokens
            token_counts.update(tokens)
            total_tokens += len(tokens)
        
        # Most common tokens
        stats['most_common_tokens'] = token_counts.most_common(50)
        stats['total_tokens_analyzed'] = total_tokens
        stats['average_tokens_per_text'] = total_tokens / len(texts[:1000])
        
        # Subword analysis
        subword_tokens = [token for token in vocab.keys() if token.startswith("##")]
        stats['num_subword_tokens'] = len(subword_tokens)
        stats['subword_ratio'] = len(subword_tokens) / len(vocab)
        
        # Coverage analysis
        vocab_tokens = set(vocab.keys())
        unique_tokens_in_corpus = set()
        for tokens in [tokenizer.encode(text).tokens for text in texts[:100]]:
            unique_tokens_in_corpus.update(tokens)
        
        stats['vocab_coverage'] = len(unique_tokens_in_corpus & vocab_tokens) / len(unique_tokens_in_corpus)
        stats['unk_ratio'] = token_counts.get('[UNK]', 0) / total_tokens
        
        # Save statistics
        stats_path = output_dir / "vocabulary_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"Statistics saved to {stats_path}")
        
        # Create summary report
        self.create_summary_report(stats, output_dir)
    
    def create_summary_report(self, stats: Dict, output_dir: Path):
        """Create a human-readable summary report."""
        report_path = output_dir / "vocabulary_report.md"
        
        report = f"""# Vocabulary Building Report

## Summary

- **Vocabulary Size**: {stats['vocab_size']:,}
- **Number of Texts**: {stats['num_texts']:,}
- **Total Tokens Analyzed**: {stats['total_tokens_analyzed']:,}
- **Average Tokens per Text**: {stats['average_tokens_per_text']:.2f}

## Token Analysis

- **Subword Tokens**: {stats['num_subword_tokens']:,} ({stats['subword_ratio']:.2%})
- **Vocabulary Coverage**: {stats['vocab_coverage']:.2%}
- **UNK Token Ratio**: {stats['unk_ratio']:.4%}

## Special Tokens

{chr(10).join(f"- {token}" for token in stats['special_tokens'])}

## Most Common Tokens

| Rank | Token | Frequency |
|------|-------|-----------|
"""
        
        for i, (token, count) in enumerate(stats['most_common_tokens'][:20], 1):
            report += f"| {i} | `{token}` | {count:,} |\n"
        
        report += f"""
## Configuration Used

```yaml
vocab_size: {self.config['vocabulary']['vocab_size']}
min_frequency: {self.config['vocabulary']['min_frequency']}
do_lower_case: {self.config['vocabulary']['tokenization']['do_lower_case']}
strip_accents: {self.config['vocabulary']['tokenization']['strip_accents']}
```

## Recommendations

"""
        
        # Add recommendations based on statistics
        if stats['unk_ratio'] > 0.05:
            report += "- ⚠️ High UNK ratio detected. Consider increasing vocabulary size or decreasing min_frequency.\n"
        else:
            report += "- ✅ UNK ratio is acceptable.\n"
        
        if stats['subword_ratio'] < 0.3:
            report += "- ⚠️ Low subword ratio. Model might struggle with out-of-vocabulary words.\n"
        else:
            report += "- ✅ Good subword token ratio for handling rare words.\n"
        
        if stats['vocab_coverage'] < 0.9:
            report += "- ⚠️ Low vocabulary coverage. Consider increasing vocabulary size.\n"
        else:
            report += "- ✅ Good vocabulary coverage.\n"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Summary report saved to {report_path}")
    
    def analyze_domain_terms(self, 
                           texts: List[str], 
                           output_dir: Path,
                           top_k: int = 1000) -> List[str]:
        """Analyze domain-specific terms in the corpus."""
        logger.info("Analyzing domain-specific terms...")
        
        # Tokenize all texts and count terms
        all_words = []
        for text in tqdm(texts, desc="Extracting words"):
            # Simple word extraction
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
            all_words.extend(words)
        
        # Count word frequencies
        word_counts = Counter(all_words)
        
        # Filter by frequency and length
        domain_terms = []
        for word, count in word_counts.most_common(top_k):
            if (count >= self.min_frequency and 
                len(word) > 2 and 
                word not in {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'may', 'she', 'use', 'your'}):
                domain_terms.append(word)
        
        # Save domain terms
        domain_terms_path = output_dir / "domain_terms.txt"
        with open(domain_terms_path, 'w', encoding='utf-8') as f:
            for term in domain_terms:
                f.write(f"{term}\n")
        
        logger.info(f"Found {len(domain_terms)} domain-specific terms")
        return domain_terms
    
    def validate_vocabulary(self, vocab_path: str, test_texts: List[str]) -> Dict:
        """Validate the built vocabulary on test texts."""
        logger.info("Validating vocabulary...")
        
        # Load tokenizer
        tokenizer = BertTokenizerFast.from_pretrained(
            str(Path(vocab_path).parent / f"{Path(vocab_path).stem}_transformers")
        )
        
        # Test tokenization
        total_tokens = 0
        unk_tokens = 0
        
        for text in test_texts:
            tokens = tokenizer.tokenize(text)
            total_tokens += len(tokens)
            unk_tokens += tokens.count('[UNK]')
        
        validation_results = {
            'total_tokens': total_tokens,
            'unk_tokens': unk_tokens,
            'unk_ratio': unk_tokens / total_tokens if total_tokens > 0 else 0,
            'vocab_size': tokenizer.vocab_size,
            'average_tokens_per_text': total_tokens / len(test_texts) if test_texts else 0
        }
        
        logger.info(f"Validation results: UNK ratio = {validation_results['unk_ratio']:.4f}")
        return validation_results
