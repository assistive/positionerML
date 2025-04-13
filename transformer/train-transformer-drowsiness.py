import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer model
    """
    def __init__(self, d_model, max_seq_length=600):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # x has shape (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]

class TransformerDrowsinessDetector(nn.Module):
    """
    Transformer-based model for drowsiness detection using IMU data
    
    Input shape: (batch_size, seq_length, n_features)
    Output: Drowsiness probability and attention weights
    """
    def __init__(self, 
                 input_dim=9,         # 9 features (time, acc_xyz, gyro_xyz, vehicle_acc_xy)
                 d_model=128,         # Embedding dimension
                 nhead=8,             # Number of attention heads
                 num_encoder_layers=3, # Number of transformer layers
                 dim_feedforward=256, # Dimension of feedforward network
                 dropout=0.1,         # Dropout rate
                 max_seq_length=600,  # Maximum sequence length (30s at 20Hz)
                 num_classes=2):      # Binary classification (drowsy/alert)
        super(TransformerDrowsinessDetector, self).__init__()
        
        # Input embedding
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Important for our input format
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Output layers
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(d_model, 64)
        self.fc2 = nn.Linear(64, 32)
        
        # Classification head
        self.classification_head = nn.Linear(32, num_classes)
        
        # Regression head (drowsiness score)
        self.regression_head = nn.Linear(32, 1)
        
        # Attention weights for visualization (feature attribution)
        self.attention_weights = None
        
    def forward(self, x, src_key_padding_mask=None):
        # x shape: (batch_size, seq_length, features)
        
        # Embed input
        x = self.embedding(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through transformer encoder
        # Store attention weights for feature attribution
        encoder_output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Global average pooling
        # Transpose to (batch_size, d_model, seq_length) for pooling
        pooled = self.global_avg_pool(encoder_output.transpose(1, 2)).squeeze(-1)
        
        # Feed-forward layers
        x = F.relu(self.fc1(pooled))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.fc2(x))
        
        # Classification output (softmax for probabilities)
        classification_output = self.classification_head(x)
        
        # Regression output (drowsiness score between 0-1)
        regression_output = torch.sigmoid(self.regression_head(x))
        
        return classification_output, regression_output, encoder_output

def train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=0.001):
    """
    Train the model using the provided data loaders
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Classification loss and regression loss
    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_cls_loss = 0.0
        train_reg_loss = 0.0
        
        for batch_idx, (data, cls_target, reg_target) in enumerate(train_loader):
            data, cls_target, reg_target = data.to(device), cls_target.to(device), reg_target.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            cls_output, reg_output, _ = model(data)
            
            # Calculate losses
            cls_loss = cls_criterion(cls_output, cls_target)
            reg_loss = reg_criterion(reg_output.squeeze(), reg_target)
            
            # Combined loss (weighted sum)
            combined_loss = cls_loss + 0.3 * reg_loss
            
            # Backward pass and optimization
            combined_loss.backward()
            optimizer.step()
            
            # Track losses
            train_cls_loss += cls_loss.item()
            train_reg_loss += reg_loss.item()
            
        # Validation phase
        model.eval()
        val_cls_loss = 0.0
        val_reg_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, cls_target, reg_target in val_loader:
                data, cls_target, reg_target = data.to(device), cls_target.to(device), reg_target.to(device)
                
                # Forward pass
                cls_output, reg_output, _ = model(data)
                
                # Calculate losses
                cls_loss = cls_criterion(cls_output, cls_target)
                reg_loss = reg_criterion(reg_output.squeeze(), reg_target)
                
                # Track losses
                val_cls_loss += cls_loss.item()
                val_reg_loss += reg_loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(cls_output.data, 1)
                total += cls_target.size(0)
                correct += (predicted == cls_target).sum().item()
        
        # Print epoch statistics
        train_cls_loss /= len(train_loader)
        train_reg_loss /= len(train_loader)
        val_cls_loss /= len(val_loader)
        val_reg_loss /= len(val_loader)
        val_acc = 100 * correct / total
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Training: Class Loss = {train_cls_loss:.4f}, Reg Loss = {train_reg_loss:.4f}')
        print(f'  Validation: Class Loss = {val_cls_loss:.4f}, Reg Loss = {val_reg_loss:.4f}, Acc = {val_acc:.2f}%')
        
        # Learning rate scheduler
        scheduler.step(val_cls_loss + 0.3 * val_reg_loss)
    
    return model

def prepare_data(data, seq_length=600, step=10):
    """
    Prepare IMU data for the model
    
    Args:
        data: DataFrame with IMU data (timestamps, accelerometer, gyroscope, etc.)
        seq_length: Number of time steps in each sequence
        step: Step size for sliding window
        
    Returns:
        List of sequences as numpy arrays
    """
    sequences = []
    labels = []
    scores = []
    
    # Extract features (normalize accelerometer and gyroscope data)
    features = [
        'timestamp_normalized',  # Normalized time within window
        'acc_x_normalized', 'acc_y_normalized', 'acc_z_normalized',
        'gyro_x_normalized', 'gyro_y_normalized', 'gyro_z_normalized',
        'vehicle_acc_x_normalized', 'vehicle_acc_y_normalized'
    ]
    
    # Sliding window approach
    for i in range(0, len(data) - seq_length + 1, step):
        seq = data.iloc[i:i+seq_length][features].values
        
        # Get label for this sequence (drowsy/alert)
        # Assuming 'drowsy' is a column in the dataset
        if 'drowsy' in data.columns:
            # Use majority vote for the label
            label = 1 if data.iloc[i:i+seq_length]['drowsy'].mean() > 0.5 else 0
        else:
            # If no label, assume test data
            label = 0
            
        # Calculate a continuous drowsiness score if available
        if 'drowsiness_score' in data.columns:
            score = data.iloc[i:i+seq_length]['drowsiness_score'].mean()
        else:
            # Default score based on label
            score = 1.0 if label == 1 else 0.0
            
        sequences.append(seq)
        labels.append(label)
        scores.append(score)
        
    return np.array(sequences), np.array(labels), np.array(scores)

def export_to_onnx(model, sample_input, output_path):
    """
    Export the PyTorch model to ONNX format for mobile deployment
    """
    torch.onnx.export(
        model,
        sample_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['classification', 'regression', 'attention'],
        dynamic_axes={'input': {0: 'batch_size', 1: 'sequence_length'},
                      'classification': {0: 'batch_size'},
                      'regression': {0: 'batch_size'},
                      'attention': {0: 'batch_size', 1: 'sequence_length'}}
    )
    print(f"Model exported to {output_path}")

def export_to_coreml(model, sample_input, output_path):
    """
    Export the PyTorch model to CoreML format for iOS deployment
    """
    try:
        import coremltools as ct
        from coremltools.models.neural_network import quantization_utils
        
        # First export to ONNX
        onnx_path = output_path.replace(".mlmodel", ".onnx")
        export_to_onnx(model, sample_input, onnx_path)
        
        # Convert ONNX to CoreML
        mlmodel = ct.converters.onnx.convert(
            model=onnx_path,
            minimum_ios_deployment_target='13'
        )
        
        # Quantize the model to reduce size
        mlmodel_quantized = quantization_utils.quantize_weights(mlmodel, 8)
        
        # Save the model
        mlmodel_quantized.save(output_path)
        print(f"Model exported to {output_path}")
        
    except ImportError:
        print("coremltools not installed. Please install with pip install coremltools")

def export_to_tflite(model, sample_input, output_path):
    """
    Export the PyTorch model to TFLite format for Android deployment
    """
    try:
        import torch.onnx
        import onnx
        import onnx_tf
        import tensorflow as tf
        
        # First export to ONNX
        onnx_path = output_path.replace(".tflite", ".onnx")
        export_to_onnx(model, sample_input, onnx_path)
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Convert ONNX to TensorFlow
        tf_rep = onnx_tf.backend.prepare(onnx_model)
        tf_path = output_path.replace(".tflite", "_tf")
        tf_rep.export_graph(tf_path)
        
        # Convert TensorFlow model to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
        
        # Enable optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert the model
        tflite_model = converter.convert()
        
        # Save the model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
            
        print(f"Model exported to {output_path}")
        
    except ImportError:
        print("Required libraries not installed. Please install onnx, onnx-tf, and tensorflow")

# Usage example
if __name__ == "__main__":
    # Create model
    model = TransformerDrowsinessDetector(
        input_dim=9,
        d_model=128,
        nhead=8,
        num_encoder_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_length=600,
        num_classes=2
    )
    
    # Print model architecture
    print(model)
    
    # Create sample input for export
    sample_input = torch.randn(1, 600, 9)
    
    # Export model to different formats
    export_to_onnx(model, sample_input, "drowsiness_detector.onnx")
    export_to_tflite(model, sample_input, "drowsiness_detector.tflite")
    export_to_coreml(model, sample_input, "drowsiness_detector.mlmodel")
