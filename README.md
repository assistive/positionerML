# Enhanced RNN for IMU-Based Dead Reckoning 

This repository provides an enhanced RNN training pipeline for IMU-based dead reckoning during GPS outages. The goal is to predict position changes from gyroscope and accelerometer data, allowing accurate position tracking for up to 60 seconds without GPS.

## Features

- ✅ RNN/LSTM model with optional bidirectionality and dropout
- ✅ Preprocessing pipeline with norm-based feature engineering
- ✅ Colab notebook for easy training and visualization
- ✅ Compatible with OxIOD and UCI HAR-style datasets
- ✅ Simulates GPS loss during training for improved robustness

## Project Files

- `train_rnn.py`: Main training script for PyTorch
- `rnn_dead_reckoning_colab.ipynb`: Google Colab notebook for interactive training
- `trained_rnn_deadreckoning.pth`: Output model weights (after training)

## How to Use

### 1. Prepare Your Data

Ensure your dataset includes the following columns:

```
time, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, dx, dy, dz
```

Save it as a CSV file.

### 2. Run the Training Script (Locally)

```bash
pip install numpy pandas torch scikit-learn
python train_rnn.py
```

Make sure to update the `DATA_PATH` variable in `train_rnn.py` with the correct path to your dataset.

### 3. Run in Google Colab

- Open `rnn_dead_reckoning_colab.ipynb`
- Upload your dataset when prompted
- Run all cells

## Comparison with 6-DOF-Inertial-Odometry

| Feature                   | Enhanced RNN Approach                             | 6-DOF-Inertial-Odometry                              |
| ------------------------- | ------------------------------------------------- | ---------------------------------------------------- |
| **Objective**             | Predict position changes (dx, dy, dz)             | Predict full 6-DOF pose (position + orientation)     |
| **Input Data**            | IMU/Gyroscope data from Android and iOS devices   | OxIOD / EuRoC MAV IMU + pose data                    |
| **Model**                 | RNN (LSTM) with bidirectional and dropout options | TensorFlow/Keras deep model for full pose estimation |
| **Dataset Compatibility** | OxIOD, UCI HAR-style (standard CSV)               | OxIOD, EuRoC MAV datasets                            |
| **Output**                | Position deltas                                   | Full 3D position + orientation                       |

This RNN-based solution is optimized for GPS-denied positioning and can be extended to 6-DOF by incorporating orientation prediction.

---

## License

MIT

## Authors

Your Name / Org


