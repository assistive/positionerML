# bert_mobile/scripts/prepare_data.py

#!/usr/bin/env python3

import argparse
import sys
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from tokenizer_utils import BERTTokenizerUtils
from transformers import BertTokenizer

class BERTDataProcessor:
    """Process and prepare data for BERT training."""
    
    def __init__(self, tokenizer_path: str, max_length: int = 512):
        try:
            if tokenizer_path.endswith('.txt'):
                # Load from vocab file
                self.tokenizer = BertTokenizer.from_pretrained(
                    'bert-base-uncased',
                    vocab_file=tokenizer_path,
                    do_lower_case=True
                )
            else:
                # Load from directory or model name
                self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            # Fallback to base tokenizer
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        self.max_length = max_length
    
    def process_text_files(self, input_dir: str, output_dir: str):
        """Process text files for language modeling."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_texts = []
        
        # Load text files
        for txt_file in input_path.rglob("*.txt"):
            with open(txt_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f.readlines() if line.strip()]
                all_texts.extend(texts)
        
        print(f"Loaded {len(all_texts)} text samples")
        
        # Process and tokenize
        processed_data = []
        for text in all_texts:
            tokens = self.tokenizer.encode_plus(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            processed_data.append({
                'text': text,
                'input_ids': tokens['input_ids'].squeeze().tolist(),
                'attention_mask': tokens['attention_mask'].squeeze().tolist()
            })
        
        # Split data
        train_size = int(0.8 * len(processed_data))
        val_size = int(0.1 * len(processed_data))
        
        train_data = processed_data[:train_size]
        val_data = processed_data[train_size:train_size + val_size]
        test_data = processed_data[train_size + val_size:]
        
        # Save processed data
        with open(output_path / "train.json", 'w') as f:
            json.dump(train_data, f, indent=2)
        
        with open(output_path / "val.json", 'w') as f:
            json.dump(val_data, f, indent=2)
        
        with open(output_path / "test.json", 'w') as f:
            json.dump(test_data, f, indent=2)
        
        print(f"Processed data saved to {output_path}")
        return str(output_path)
    
    def process_classification_data(self, csv_path: str, output_dir: str, 
                                  text_column: str, label_column: str):
        """Process CSV data for classification."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load CSV
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} samples")
        
        # Process data
        processed_data = []
        label_map = {label: idx for idx, label in enumerate(df[label_column].unique())}
        
        for _, row in df.iterrows():
            text = str(row[text_column])
            label = label_map[row[label_column]]
            
            tokens = self.tokenizer.encode_plus(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            processed_data.append({
                'text': text,
                'label': label,
                'input_ids': tokens['input_ids'].squeeze().tolist(),
                'attention_mask': tokens['attention_mask'].squeeze().tolist()
            })
        
        # Split data
        train_data, temp_data = train_test_split(processed_data, test_size=0.3, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        
        # Save processed data
        with open(output_path / "train.json", 'w') as f:
            json.dump(train_data, f, indent=2)
        
        with open(output_path / "val.json", 'w') as f:
            json.dump(val_data, f, indent=2)
        
        with open(output_path / "test.json", 'w') as f:
            json.dump(test_data, f, indent=2)
        
        # Save label mapping
        with open(output_path / "label_map.json", 'w') as f:
            json.dump(label_map, f, indent=2)
        
        print(f"Classification data processed and saved to {output_path}")
        print(f"Label mapping: {label_map}")
        return str(output_path)

def main():
    parser = argparse.ArgumentParser(description='Prepare data for BERT training')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory or CSV file')
    parser.add_argument('--vocab_path', type=str, required=True,
                       help='Path to vocabulary/tokenizer')
    parser.add_argument('--output_dir', type=str, default='./data/processed',
                       help='Output directory for processed data')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--task_type', type=str, choices=['classification', 'language_modeling'],
                       default='language_modeling', help='Type of task')
    parser.add_argument('--text_column', type=str, default='text',
                       help='Name of text column (for classification)')
    parser.add_argument('--label_column', type=str, default='label',
                       help='Name of label column (for classification)')
    
    args = parser.parse_args()
    
    try:
        print("Processing data for BERT training...")
        
        # Initialize processor
        processor = BERTDataProcessor(args.vocab_path, args.max_length)
        
        # Process based on task type
        if args.task_type == 'classification':
            if not args.input_dir.endswith('.csv'):
                raise ValueError("Classification task requires CSV input file")
            
            output_path = processor.process_classification_data(
                args.input_dir, args.output_dir, 
                args.text_column, args.label_column
            )
        else:
            output_path = processor.process_text_files(
                args.input_dir, args.output_dir
            )
        
        print("Data preparation completed successfully!")
        print(f"Processed data saved to: {output_path}")
        
    except Exception as e:
        print(f"Error preparing data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

