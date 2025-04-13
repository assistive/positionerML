import torch
import torch.nn as nn
import torch.nn.functional as F

class IMUTransformer(nn.Module):
    def __init__(self, input_dim=8, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, 
                 dropout=0.1, max_seq_length=200, num_classes=2):
        super(IMUTransformer, self).__init__()
        
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Two separate task heads
        self.accident_classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # Binary classification for accident detection
        )
        
        self.drowsiness_classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)  # 3 classes: alert, slightly drowsy, severely drowsy
        )
        
    def forward(self, x, task='accident'):
        # x shape: [batch_size, seq_length, input_dim]
        x = self.embedding(x)  # [batch_size, seq_length, d_model]
        x = self.pos_encoder(x)
        
        # Transformer expects [seq_length, batch_size, d_model]
        x = x.permute(1, 0, 2)
        
        # Apply transformer encoder
        transformer_output = self.transformer_encoder(x)
        
        # Use the last token's representation for classification
        seq_representation = transformer_output[-1, :, :]
        
        # Choose classifier based on task
        if task == 'accident':
            return self.accident_classifier(seq_representation)
        elif task == 'drowsiness':
            return self.drowsiness_classifier(seq_representation)
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
