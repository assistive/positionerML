# SpatialLM

SpatialLM is a transformer-based language model enhanced with spatial understanding capabilities. It extends traditional language models by incorporating spatial context and enabling spatial reasoning in natural language processing tasks.

![SpatialLM Architecture](https://github.com/user-attachments/assets/d16d701f-c757-47c5-9732-da61877da64a)

## Features

- 🔄 **Bidirectional Spatial Context**: Process language with spatial context in a bidirectional manner
- 📍 **Spatial Coordinates Integration**: Incorporate spatial coordinates (x, y, z) into language representations
- 🧠 **Spatial Reasoning**: Generate text with awareness of spatial relationships
- 🏙️ **Location Understanding**: Comprehend and reason about locations, distances, and spatial arrangements
- 📱 **Mobile Deployment**: Optimized for deployment on iOS and Android devices

## Models

SpatialLM is available in different sizes:

| Model | Parameters | Description |
|-------|------------|-------------|
| SpatialLM-Small | 70M | Lightweight model for mobile applications |
| SpatialLM-Base | 220M | Balanced model for most applications |
| SpatialLM-Large | 770M | Most powerful model for complex spatial reasoning |

## Setup and Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- Transformers 4.25+
- For iOS deployment: Xcode 13+, macOS Monterey+
- For Android deployment: Android Studio Arctic Fox+, NDK 21+

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/spatialLM.git
cd spatialLM

# Install dependencies
pip install -r requirements.txt
```

### Downloading Pre-trained Models

Use the provided script to download pre-trained models:

```bash
python training/download_model.py --model_name spatialLM-base --output_dir ./models
```

Available model options: `spatialLM-small`, `spatialLM-base`, `spatialLM-large`.

## Training

### Data Preparation

To train SpatialLM, you need data with spatial coordinates. The expected format is a CSV or JSON file with the following columns:

- `text`: The text content
- `x`, `y`, `z`: Spatial coordinates

Example data format:

```csv
text,x,y,z
"The car is parked at the corner of Main St. and Broadway.",42.3601,-71.0589,0
"The coffee shop is located across from the library.",42.3608,-71.0593,0
```

### Training the Model

```bash
python training/train.py \
    --base_model gpt2 \
    --spatial_mode \
    --train_file path/to/your/data.csv \
    --coordinate_columns x,y,z \
    --output_dir ./trained_models \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --learning_rate 5e-5
```

### Fine-tuning

To fine-tune a pre-trained SpatialLM model:

```bash
python training/finetune.py \
    --model_path ./models/spatialLM-base \
    --train_file path/to/your/data.csv \
    --output_dir ./finetuned_models \
    --finetune_method lora \
    --num_train_epochs 2
```

## Evaluation

Evaluate your trained or fine-tuned model:

```bash
python training/evaluate.py \
    --model_path ./trained_models/your_model \
    --test_file path/to/test_data.csv
```

## Deployment

### iOS Deployment

1. Convert the model to Core ML format:

```bash
python deployment/ios/conversion_scripts/convert_to_coreml.py \
    --model_path ./trained_models/your_model \
    --output_dir ./coreml_models \
    --include_tokenizer
```

2. Follow the instructions in the [iOS Integration Guide](deployment/ios/integration_guide.md) to integrate the model into your iOS application.

### Android Deployment

1. Convert the model to TensorFlow Lite format:

```bash
python deployment/android/conversion_scripts/convert_to_tflite.py \
    --model_path ./trained_models/your_model \
    --output_dir ./tflite_models \
    --quantize \
    --include_tokenizer
```

2. Follow the instructions in the [Android Integration Guide](deployment/android/integration_guide.md) to integrate the model into your Android application.

## Model Architecture

SpatialLM extends traditional transformer-based language models with several spatial components:

1. **Spatial Embeddings**: Transform spatial coordinates into embeddings that can be fused with language representations
2. **Spatial Attention**: Attention mechanism that allows the model to focus on relevant spatial information
3. **Spatial Head**: Specialized head for predicting spatial coordinates from language representations

The architecture can be visualized as follows:

```
Input Text + Spatial Coordinates
       ↓
Input Embeddings + Spatial Embeddings
       ↓
Transformer Layers with Spatial Attention
       ↓
     Outputs
       ↓
  ┌────┴────┐
  ↓         ↓
Language   Spatial
 Head      Head
```

## Use Cases

SpatialLM is designed for applications that require understanding spatial relationships in language:

- 🗺️ **Location-based services**: Generate descriptions of places based on coordinates
- 🤖 **Robotics**: Natural language instructions with spatial context
- 🏙️ **Urban planning**: Analyzing spatial descriptions of urban environments
- 🎮 **Gaming**: NPC dialogues with spatial awareness
- 🏥 **Healthcare**: Describing medical images with spatial context
- 🚘 **Autonomous vehicles**: Natural language interaction with spatial understanding

## Examples

### Python Inference

```python
from transformers import AutoTokenizer
from models.spatialLM import SpatialLM

# Load model and tokenizer
model = SpatialLM.from_pretrained("./models/spatialLM-base")
tokenizer = AutoTokenizer.from_pretrained("./models/spatialLM-base")

# Predict spatial coordinates from text
text = "The coffee shop is on the corner of Main St. and Broadway."
coordinates = model.predict_spatial(text, tokenizer)
print(f"Predicted coordinates: {coordinates}")

# Generate text with spatial context
prompt = "Describe the location at coordinates [42.3601, -71.0589, 0]:"
spatial_coordinates = [42.3601, -71.0589, 0]
generated_text = model.generate(
    prompt,
    tokenizer,
    spatial_coordinates=spatial_coordinates
)
print(f"Generated text: {generated_text}")
```

### API Usage

If you're using our API service:

```python
import requests

# Predict spatial coordinates
response = requests.post(
    "https://api.spatialLM.ai/predict",
    json={"text": "The restaurant is across from the library."}
)
print(response.json())  # {"x": 42.3601, "y": -71.0589, "z": 0}

# Generate text with spatial context
response = requests.post(
    "https://api.spatialLM.ai/generate",
    json={
        "prompt": "Describe this location:",
        "coordinates": [42.3601, -71.0589, 0]
    }
)
print(response.json())  # {"text": "This location is in downtown Boston..."}
```

## Contributing

We welcome contributions to the SpatialLM project! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use SpatialLM in your research, please cite:

```bibtex
@article{spatialLM2025,
  title={SpatialLM: A Language Model with Spatial Understanding Capabilities},
  author={Smith, John and Doe, Jane},
  journal={Proceedings of the Conference on Language Models with Spatial Understanding},
  year={2025}
}
```

## Acknowledgments

- The project builds upon the work of the Hugging Face Transformers library
- Special thanks to all contributors and the open-source community
