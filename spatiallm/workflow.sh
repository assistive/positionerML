# 1. Download the pre-trained model (KEEP)
python training/download_model.py --model_name spatialLM-1.1-qwen-0.5b

# 2. Test the model (NEW - KEEP)
python scripts/test_spatiallm.py

# 3. Fine-tune if needed (KEEP - simplified)
python training/finetune.py --model_path ./models/spatialLM-1.1-qwen-0.5b

# 4. Convert for mobile (KEEP - updated)
python ios/convert_to_coreml.py --model_path ./models/spatialLM-1.1-qwen-0.5b
python android/convert_to_tflite.py --model_path ./models/spatialLM-1.1-qwen-0.5b
