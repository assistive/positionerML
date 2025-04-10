# Filename: train_rnn.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import random
import os

# --- Config ---
WINDOW_SIZE = 100
STRIDE = 50
USE_BIDIRECTIONAL = True
DROPOUT = 0.3
HIDDEN_SIZE = 128
NUM_LAYERS = 2
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
GPS_OUTAGE_PROB = 0.3
VALIDATION_SPLIT = 0.2
DATA_PATH = "path_to_your_dataset.csv"  # Replace with your actual dataset path

# --- Dataset Loader ---
class DeadReckoningDataset(Dataset):
    def __init__(self, df):
        df = df.copy()
        imu_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

        df['acc_mag'] = np.linalg.norm(df[['acc_x', 'acc_y', 'acc_z']].values, axis=1)
        df['gyro_mag'] = np.linalg.norm(df[['gyro_x', 'gyro_y', 'gyro_z']].values, axis=1)

        features = df[imu_cols + ['acc_mag', 'gyro_mag']].values
        targets = df[['dx', 'dy', 'dz']].values

        self.X, self.y = [], []
        for i in range(0, len(df) - WINDOW_SIZE, STRIDE):
            self.X.append(features[i:i+WINDOW_SIZE])
            self.y.append(targets[i+WINDOW_SIZE-1])

        self.X = np.array(self.X)
        self.y = np.array(self.y)

        self.scaler = StandardScaler()
        n_samples, window, n_features = self.X.shape
        self.X = self.scaler.fit_transform(self.X.reshape(-1, n_features)).reshape(n_samples, window, n_features)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# --- Model Definition ---
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout, bidirectional=bidirectional)
        direction = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction, 3)

    def forward(self, x):
        _, (h_n, _) = self.rnn(x)
        if self.rnn.bidirectional:
            h_out = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            h_out = h_n[-1]
        return self.fc(h_out)

# --- Utilities ---
def rmse(preds, targets):
    return torch.sqrt(nn.MSELoss()(preds, targets))

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        if random.random() < GPS_OUTAGE_PROB:
            y_batch = torch.zeros_like(y_batch)  # Simulate GPS loss
        X_batch, y_batch = X_batch.float(), y_batch.float()
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader):
    model.eval()
    losses = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.float(), y_batch.float()
            outputs = model(X_batch)
            losses.append(rmse(outputs, y_batch).item())
    return np.mean(losses)

# --- Training Pipeline ---
def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    dataset = DeadReckoningDataset(df)

    val_size = int(len(dataset) * VALIDATION_SPLIT)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = RNNModel(input_size=8, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                     dropout=DROPOUT, bidirectional=USE_BIDIRECTIONAL)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        loss = train(model, train_loader, optimizer, criterion)
        val_rmse = validate(model, val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {loss:.4f} - Val RMSE: {val_rmse:.4f}")

    torch.save(model.state_dict(), "trained_rnn_deadreckoning.pth")
    print("✅ Model saved to trained_rnn_deadreckoning.pth")

if __name__ == "__main__":
    main()

