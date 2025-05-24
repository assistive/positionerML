# bert_mobile/scripts/build_vocabulary.py

#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from vocab_builder import VocabularyBuilder

def main():
    parser = argparse.ArgumentParser(description='Build custom vocabulary for BERT')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing text files')
    parser.add_argument('--output_dir', type=str, default='./data/vocabularies',
                       help='Output directory for vocabulary files')
    parser.add_argument('--vocab_size', type=int, default=30000,
                       help='Target vocabulary size')
    parser.add_argument('--min_frequency', type=int, default=5,
                       help='Minimum token frequency')
    parser.add_argument('--vocab_name', type=str, default='custom_vocab',
                       help='Name for the vocabulary')
    parser.add_argument('--config', type=str, default='config/vocab_config.yaml',
                       help='Path to vocabulary configuration')
    parser.add_argument('--analyze_domain', action='store_true',
                       help='Analyze domain-specific terms')
    parser.add_argument('--validate', action='store_true',
                       help='Validate vocabulary with test texts')
    parser.add_argument('--test_texts', type=str,
                       help='Path to test texts for validation')
    
    args = parser.parse_args()
    
    try:
        print("Building custom vocabulary...")
        
        # Initialize vocabulary builder
        builder = VocabularyBuilder(args.config)
        
        # Override config with command line arguments
        builder.vocab_size = args.vocab_size
        builder.min_frequency = args.min_frequency
        
        # Get corpus files
        input_path = Path(args.input_dir)
        if input_path.is_file():
            corpus_paths = [str(input_path)]
        else:
            corpus_paths = [str(input_path)]
        
        # Build vocabulary
        vocab_path = builder.build_from_corpus(
            corpus_paths=corpus_paths,
            output_dir=args.output_dir,
            vocab_name=args.vocab_name
        )
        
        print(f"Vocabulary built successfully: {vocab_path}")
        
        # Analyze domain terms if requested
        if args.analyze_domain:
            texts = builder.load_texts(corpus_paths)
            domain_terms = builder.analyze_domain_terms(
                texts, Path(args.output_dir), top_k=1000
            )
            print(f"Found {len(domain_terms)} domain-specific terms")
        
        # Validate vocabulary if requested
        if args.validate:
            if args.test_texts:
                with open(args.test_texts, 'r') as f:
                    test_texts = [line.strip() for line in f.readlines() if line.strip()]
            else:
                # Use subset of training texts for validation
                test_texts = builder.load_texts(corpus_paths)[:100]
            
            validation_results = builder.validate_vocabulary(vocab_path, test_texts)
            print(f"\nVocabulary Validation Results:")
            print(f"  UNK ratio: {validation_results['unk_ratio']:.4f}")
            print(f"  Average tokens per text: {validation_results['average_tokens_per_text']:.2f}")
            
            if validation_results['unk_ratio'] > 0.05:
                print("  ⚠️ High UNK ratio - consider increasing vocabulary size")
            else:
                print("  ✅ UNK ratio acceptable")
        
        print("Vocabulary building completed!")
        
    except Exception as e:
        print(f"Error building vocabulary: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


