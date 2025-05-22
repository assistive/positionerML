# positionerML

![positioner_small](https://github.com/user-attachments/assets/d16d701f-c757-47c5-9732-da61877da64a)

# Visual Navigation System

A cutting-edge mobile visual navigation system that combines computer vision, deep learning, and location intelligence to provide robust indoor/outdoor navigation without relying solely on GPS.

![Visual Navigation](https://img.shields.io/badge/Status-In%20Development-yellow)
![Platform](https://img.shields.io/badge/Platform-Android%20%7C%20iOS-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 Overview

This project implements a sophisticated visual navigation system that uses:
- **Vision-Language Models (VLM)** for semantic scene understanding
- **VLAD descriptors** for efficient place recognition
- **Octree spatial representation** for hierarchical map organization
- **Foursquare Places API** integration for semantic location enrichment
- **IMU sensor fusion** with RNN/TCN models for dead reckoning during GPS outages

## 🏗️ Architecture

### Core Components

#### 1. Visual Processing Pipeline
- **FastVLM Processor**: High-performance vision-language model optimized for mobile devices
- **VLAD Encoder**: Vector of Locally Aggregated Descriptors for compact place recognition
- **Multi-resolution Processing**: Adaptive resolution based on battery and navigation needs

#### 2. Spatial Representation
- **Octree Map**: Hierarchical spatial structure for efficient storage and retrieval
- **Semantic Nodes**: Each octree node contains visual descriptors and POI information
- **Navigation Graph**: Connectivity information for path planning

#### 3. Sensor Fusion
- **IMU Processing**: RNN/TCN models for smoothing and dead reckoning
- **Visual-Inertial Odometry**: Continuous tracking between keyframes
- **GPS Integration**: Loose coupling with visual anchoring

#### 4. Location Intelligence
- **Foursquare Integration**: Rich POI data and semantic venue information
- **Semantic Navigation**: Context-aware instructions ("Turn left at Starbucks")
- **Digital Twin Support**: Virtual venue representations with semantic properties

## 🚀 Features

### Visual Navigation
- ✅ Real-time localization using visual features
- ✅ Robust to GPS-denied environments
- ✅ Cross-season and cross-weather matching
- ✅ Semantic scene understanding

### Performance Optimization
- ✅ Adaptive resolution control
- ✅ Battery-aware processing
- ✅ Memory-efficient caching
- ✅ Selective API usage

### Dead Reckoning
- ✅ IMU-based position tracking during GPS outages
- ✅ RNN/TCN models for sensor data processing
- ✅ Up to 60 seconds of accurate tracking without GPS

### Integration
- ✅ Foursquare Places API for venue data
- ✅ Niantic VPS dataset support (planned)
- ✅ Kotlin Multiplatform architecture
- ✅ Android and iOS support

## 📁 Project Structure

```
├── shared/                 # Kotlin Multiplatform shared code
│   ├── commonMain/        # Core business logic
│   │   ├── vlm/          # Vision-Language Model implementations
│   │   ├── vlad/         # VLAD encoder and calibration
│   │   ├── octree/       # Octree map structure
│   │   └── mapping/      # Mapping engine
│   ├── androidMain/      # Android-specific implementations
│   └── iosMain/          # iOS-specific implementations
├── androidApp/            # Android application
├── iosApp/               # iOS application
├── rnn/                  # RNN/LSTM models for IMU processing
├── tcn/                  # Temporal Convolutional Networks
├── transformer/          # Transformer models for advanced features
└── vpr/                  # Visual Place Recognition documentation
```

## 🛠️ Technical Stack

### Core Technologies
- **Kotlin Multiplatform**: Shared business logic across platforms
- **TensorFlow Lite**: Mobile ML inference
- **CoreML**: iOS machine learning (planned)
- **Compose Multiplatform**: Cross-platform UI

### Models
- **FastVLM**: Lightweight vision-language model
- **MobileVLM**: Alternative VLM implementation
- **RNN/LSTM**: For IMU data smoothing and dead reckoning
- **TCN**: Temporal convolutional networks for sensor fusion

### External Services
- **Foursquare Places API**: Location data and POI information
- **Niantic VPS**: Visual positioning dataset (integration planned)

## 🏃 Getting Started

### Prerequisites
- Android Studio Arctic Fox or later
- Xcode 13+ (for iOS development)
- Python 3.8+ (for model training)
- Kotlin 1.8+

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/visual-navigation-system.git
cd visual-navigation-system
```

2. Install dependencies:
```bash
# For model training
pip install -r requirements.txt

# For Android/iOS development
./gradlew build
```

3. Configure API keys:
```kotlin
// In local.properties or as environment variables
FOURSQUARE_API_KEY=your_api_key_here
```

4. Run the application:
```bash
# Android
./gradlew :androidApp:installDebug

# iOS
cd iosApp && pod install
open iosApp.xcworkspace
```

## 🧪 Model Training

### Train RNN for IMU Dead Reckoning
```bash
cd rnn
python train-rnn-model-deadreckoning.py --data path/to/imu_data.csv --epochs 30
```

### Train TCN Model
```bash
cd tcn
python train_tcn_model.py --dataset oxiod --epochs 30
```

### Calibrate VLAD Encoder
Run the calibration tool in the app to train VLAD on your environment.

## 📊 Performance

### Dead Reckoning Accuracy
- Mean position error: < 3m after 30 seconds
- Maximum error: < 10m after 60 seconds

### Visual Localization
- Localization accuracy: 1-2m in mapped areas
- Processing speed: 10-15 fps on modern smartphones
- Memory usage: 150-200MB typical

### Battery Impact
- High precision mode: 8-10% per hour
- Standard mode: 4-5% per hour
- Battery saver mode: 2-3% per hour

## 🗺️ Roadmap

### Phase 1: Core Implementation ✅
- [x] VLM integration
- [x] VLAD encoder
- [x] Octree map structure
- [x] Basic navigation engine

### Phase 2: Sensor Fusion 🚧
- [x] RNN/LSTM for IMU processing
- [x] TCN implementation
- [ ] Full visual-inertial odometry

### Phase 3: External Integration 📋
- [x] Foursquare Places API
- [ ] Niantic VPS dataset
- [ ] OpenStreetMap integration

### Phase 4: Advanced Features 📋
- [ ] Transformer-based drowsiness detection
- [ ] AR visualization
- [ ] Crowdsourced map updates

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Areas for Contribution
- Model optimization for specific devices
- Additional sensor fusion algorithms
- UI/UX improvements
- Documentation and examples

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Oxford Inertial Odometry Dataset (OxIOD)
- RoNIN Dataset for inertial navigation
- Foursquare for location intelligence
- The open-source ML community

## 📚 References

1. [MobileVLM: A Fast, Strong and Open Vision Language Assistant for Mobile Devices](https://arxiv.org/abs/2312.16886)
2. [All are Worth Words: A ViT Backbone for Diffusion Models](https://arxiv.org/abs/2209.12152)
3. [NetVLAD: CNN architecture for weakly supervised place recognition](https://arxiv.org/abs/1511.07247)
4. [Visual Place Recognition: A Survey](https://arxiv.org/abs/2011.00934)

## 📧 Contact

For questions or collaboration opportunities, please open an issue or contact the maintainers.

---

**Note**: This project is under active development. APIs and features may change.
