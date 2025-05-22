# internvl/README.md

# InternVL Training, Fine-tuning, and Deployment

This folder contains the complete infrastructure for training, fine-tuning, and deploying InternVL models for mobile applications.

## Features

- ✅ Automatic model downloading from Hugging Face
- ✅ Fine-tuning pipeline with LoRA/QLoRA support
- ✅ iOS deployment via CoreML conversion
- ✅ Android deployment via TensorFlow Lite conversion
- ✅ Vision-Language understanding for IMU and driving context
- ✅ Multi-modal data processing pipeline
- ✅ Mobile optimization and quantization

## Project Structure

```
internvl/
├── README.md
├── requirements.txt
├── config/
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── deployment_config.yaml
├── src/
│   ├── __init__.py
│   ├── model_downloader.py
│   ├── data_processor.py
│   ├── trainer.py
│   ├── fine_tuner.py
│   ├── mobile_converter.py
│   └── deployment_utils.py
├── scripts/
│   ├── download_model.py
│   ├── prepare_data.py
│   ├── train_model.py
│   ├── fine_tune.py
│   ├── convert_to_mobile.py
│   └── deploy.py
├── data/
│   ├── training/
│   ├── validation/
│   └── test/
├── models/
│   ├── pretrained/
│   ├── fine_tuned/
│   └── mobile/
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_analysis.ipynb
│   └── deployment_testing.ipynb
└── tests/
    ├── test_model_download.py
    ├── test_training.py
    └── test_deployment.py
```

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Model**:
   ```bash
   python scripts/download_model.py --model internvl2-2b
   ```

3. **Prepare Data**:
   ```bash
   python scripts/prepare_data.py --data_dir ./data/raw
   ```

4. **Fine-tune Model**:
   ```bash
   python scripts/fine_tune.py --config config/training_config.yaml
   ```

5. **Convert for Mobile**:
   ```bash
   python scripts/convert_to_mobile.py --platform ios android
   ```

6. **Deploy**:
   ```bash
   python scripts/deploy.py --platform ios --output ./models/mobile/
   ```

## Configuration

Edit the YAML files in the `config/` directory to customize:
- Model parameters
- Training hyperparameters
- Mobile optimization settings
- Deployment targets

---

