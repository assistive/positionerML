# SpatialLM 1.1 - Correct File Structure and Setup

This guide shows the correct file placement and how to set up the repository properly.

## 📁 **Correct File Structure**

Based on the current repository, here's where files should be placed:

```
spatiallm/                           # Root directory
├── README.md                        # ✅ Keep existing
├── requirements.txt                 # ✅ Create/update (see below)
├── download_model.py               # 🆕 PLACE HERE (from artifacts)
├── test_model.py                   # 🆕 PLACE HERE (from artifacts)  
├── convert_to_coreml.py            # 🆕 PLACE HERE (from artifacts)
├── convert_to_tflite.py            # 🆕 PLACE HERE (from artifacts)
├── models/                         # 📁 Downloaded models go here
│   └── spatiallm-1.1-qwen-0.5b/   # 📁 Model will be downloaded here
├── coreml_models/                  # 📁 iOS converted models
├── tflite_models/                  # 📁 Android converted models
└── training/                       # 📁 Keep existing training scripts
    ├── config.py                   # ✅ Keep/update for fine-tuning only
    ├── finetune.py                 # ✅ Keep for fine-tuning
    └── prepare_data.py             # ❌ Can remove (not needed for pre-trained)
```

## 🔧 **Setup Instructions**

### 1. **Install Dependencies**

Create/update `requirements.txt` in the root `spatiallm/` directory:

```txt
# Core dependencies for SpatialLM 1.1
torch>=2.0.0
transformers>=4.35.0
huggingface-hub>=0.16.0

# Model conversion dependencies
# iOS (macOS only)
coremltools>=7.0; sys_platform == "darwin"

# Android
tensorflow>=2.13.0
onnx>=1.14.0
onnx-tf>=1.10.0

# Utilities
numpy>=1.24.0
tqdm>=4.65.0
psutil>=5.9.0
```

Install dependencies:
```bash
cd spatiallm
pip install -r requirements.txt
```

### 2. **Download the Model**

```bash
cd spatiallm
python download_model.py --model_name spatiallm-1.1-qwen-0.5b
```

### 3. **Test the Model**

```bash
python test_model.py --model_path ./models/spatiallm-1.1-qwen-0.5b
```

### 4. **Convert for Mobile**

**For iOS (macOS only):**
```bash
python convert_to_coreml.py --model_path ./models/spatiallm-1.1-qwen-0.5b
```

**For Android:**
```bash
python convert_to_tflite.py --model_path ./models/spatiallm-1.1-qwen-0.5b
```

## 🚨 **Files to Remove/Ignore**

Since we're using the pre-trained model, these files are no longer needed:

### **Can be Removed:**
- `training/train.py` - Full training not needed
- `training/prepare_data.py` - Data prep not needed  
- `models/spatialLM.py` - Custom model not needed
- `models/layers.py` - Custom layers not needed
- `utils/` directory - Training utilities not needed

### **Keep for Fine-tuning (Optional):**
- `training/finetune.py` - If you want to fine-tune
- `training/config.py` - Configuration for fine-tuning

## 📱 **Mobile Integration**

### **iOS Integration:**
1. Find your converted model in: `coreml_models/`
2. Drag the `.mlpackage` file into your Xcode project
3. Use the tokenizer files for text preprocessing
4. Check `deployment_info.json` for integration details

### **Android Integration:**
1. Copy the `.tflite` file from `tflite_models/` to your Android project's `assets/` folder
2. Use the generated `SpatialLMInference.kt` file as a starting point
3. Add the dependencies from `build.gradle.dependencies`
4. Use tokenizer files for text preprocessing

## 🔍 **Troubleshooting**

### **Common Issues:**

1. **"Module not found" errors:**
   ```bash
   pip install -r requirements.txt
   ```

2. **CoreML conversion fails:**
   - Ensure you're on macOS
   - Install: `pip install coremltools`

3. **TensorFlow Lite conversion fails:**
   - Install: `pip install tensorflow onnx onnx-tf`
   - Note: TFLite conversion of large models is challenging

4. **Model download fails:**
   - Check internet connection
   - Ensure sufficient disk space (~2GB)

### **Platform-Specific Notes:**

- **macOS:** Full functionality (CoreML + TFLite)
- **Linux:** TensorFlow Lite conversion only
- **Windows:** TensorFlow Lite conversion only

## 🎯 **Simplified Workflow**

The new workflow is much simpler:

```bash
# 1. Download pre-trained model
python download_model.py

# 2. Test it works
python test_model.py

# 3. Convert for your platform
python convert_to_coreml.py    # iOS
python convert_to_tflite.py    # Android

# 4. Integrate into your app
```

No more training from scratch, data preparation, or custom model architectures!

## 📖 **What's Different from Original Structure**

**Original (Complex):**
- Custom model training
- Data preparation pipelines  
- Multiple model variants
- Complex deployment scripts

**New (Simplified):**
- Pre-trained model download
- Direct mobile conversion
- Ready-to-use mobile integration
- Focused on deployment, not training

This makes the codebase much more maintainable and easier to use!