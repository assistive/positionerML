# BERT Mobile Training and Deployment

A comprehensive framework for training, fine-tuning, and deploying BERT models optimized for mobile devices with custom vocabularies.

## Features

- 🔧 **Custom Vocabulary Building**: Create domain-specific vocabularies from your text data
- 📱 **Mobile Optimization**: Convert BERT models for iOS (CoreML) and Android (TensorFlow Lite)
- 🎯 **Fine-tuning Support**: Fine-tune pre-trained BERT models on custom datasets
- ⚡ **Performance Optimization**: Quantization, pruning, and mobile-specific optimizations
- 🛠️ **Complete Pipeline**: End-to-end workflow from data to deployment
- 📊 **Evaluation Tools**: Comprehensive model evaluation and performance benchmarking

## Quick Start

### 1. Installation

```bash
git clone <repository-url>
cd bert_mobile
pip install -r requirements.txt
```

### 2. Build Custom Vocabulary

```bash
# Build vocabulary from your text data
python scripts/build_vocabulary.py \
    --input_dir ./data/raw \
    --output_dir ./data/vocabularies \
    --vocab_size 30000 \
    --min_frequency 5
```

### 3. Download and Prepare Base Model

```bash
# Download BERT base model
python scripts/download_model.py \
    --model_name bert-base-uncased \
    --output_dir ./models/pretrained
```

### 4. Prepare Training Data

```bash
# Process and tokenize your data
python scripts/prepare_data.py \
    --input_dir ./data/raw \
    --vocab_path ./data/vocabularies/vocab.txt \
    --output_dir ./data/processed \
    --max_length 512
```

### 5. Train/Fine-tune Model

```bash
# Fine-tune BERT on your data
python scripts/fine_tune_bert.py \
    --model_path ./models/pretrained/bert-base-uncased \
    --data_path ./data/processed \
    --vocab_path ./data/vocabularies/vocab.txt \
    --output_dir ./models/fine_tuned \
    --num_epochs 3
```

### 6. Convert for Mobile

```bash
# Convert to mobile formats
python scripts/convert_to_mobile.py \
    --model_path ./models/fine_tuned \
    --output_dir ./models/mobile \
    --platforms ios android \
    --quantize
```

## Project Structure

```
bert_mobile/
├── config/           # Configuration files
├── src/             # Core source code
├── scripts/         # Training and deployment scripts
├── data/            # Data directories
├── models/          # Model storage
├── notebooks/       # Jupyter notebooks for analysis
├── tests/           # Unit tests
└── deployment/      # Mobile deployment resources
```

## Configuration

Edit the YAML files in `config/` to customize:

- **model_config.yaml**: Model architecture and parameters
- **training_config.yaml**: Training hyperparameters
- **mobile_config.yaml**: Mobile optimization settings
- **vocab_config.yaml**: Vocabulary building parameters

## Advanced Usage

### Custom Vocabulary for Domain-Specific Tasks

```python
from src.vocab_builder import VocabularyBuilder

# Build specialized vocabulary
vocab_builder = VocabularyBuilder()
vocab_builder.build_from_corpus(
    corpus_path="./data/raw/domain_texts.txt",
    vocab_size=25000,
    special_tokens=["[DOMAIN]", "[ENTITY]"]
)
```

### Mobile-Optimized Training

```python
from src.bert_trainer import BERTMobileTrainer

trainer = BERTMobileTrainer(
    model_name="bert-base-uncased",
    vocab_path="./data/vocabularies/custom_vocab.txt",
    mobile_optimized=True
)
trainer.train(data_path="./data/processed")
```

### Performance Benchmarking

```bash
python scripts/evaluate_model.py \
    --model_path ./models/mobile/bert_mobile.tflite \
    --test_data ./data/test \
    --platform android \
    --benchmark
```

## Supported Platforms

- **iOS**: CoreML format with Neural Engine optimization
- **Android**: TensorFlow Lite with GPU/NNAPI support
- **Web**: TensorFlow.js format (experimental)

## Performance Targets

| Platform | Model Size | Inference Time | Memory Usage |
|----------|------------|----------------|--------------|
| iOS      | < 50MB     | < 100ms        | < 200MB      |
| Android  | < 50MB     | < 150ms        | < 250MB      |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{bert_mobile_2025,
  title={BERT Mobile: Training and Deployment Framework},
  author={Your Name},
  year={2025},
  url={https://github.com/your-org/bert-mobile}
}
```
