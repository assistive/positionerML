import tensorflow as tf
import numpy as np

def create_transformer_drowsiness_model(sequence_length=600, feature_dim=10):
    # Input layer
    inputs = tf.keras.layers.Input(shape=(sequence_length, feature_dim))
    
    # Positional encoding
    pos_encoding = positional_encoding(sequence_length, feature_dim)
    x = inputs + pos_encoding
    
    # Transformer encoder blocks
    x = transformer_encoder_block(x, feature_dim, num_heads=4)
    x = transformer_encoder_block(x, feature_dim, num_heads=4)
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    
    # Output: drowsiness score
    drowsiness_score = tf.keras.layers.Dense(1, activation='sigmoid', name='drowsiness_score')(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=[drowsiness_score])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def positional_encoding(length, depth):
    # Simple fixed positional encoding
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth
    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates
    
    pos_encoding = np.concatenate(
        [np.sin(angle_rads[:, 0::2]), np.cos(angle_rads[:, 1::2])],
        axis=-1)
    
    return tf.cast(pos_encoding, dtype=tf.float32)

def transformer_encoder_block(inputs, embed_dim, num_heads):
    # Multi-head attention
    attention_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim // num_heads
    )(inputs, inputs)
    
    # Skip connection 1
    attention_output = tf.keras.layers.LayerNormalization()(inputs + attention_output)
    
    # Feed forward network
    ffn_output = tf.keras.Sequential([
        tf.keras.layers.Dense(embed_dim * 2, activation='relu'),
        tf.keras.layers.Dense(embed_dim)
    ])(attention_output)
    
    # Skip connection 2
    encoder_output = tf.keras.layers.LayerNormalization()(attention_output + ffn_output)
    
    return encoder_output

# Create and export the model
model = create_transformer_drowsiness_model()

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model
with open('transformer_drowsiness_detector.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model saved as transformer_drowsiness_detector.tflite")
