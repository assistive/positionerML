import coremltools as ct

# Load the Keras model
model = create_transformer_drowsiness_model()

# Input description
input_description = {
    "input_sequence": "Sequence of motion data features",
    "sequenceLength": "Length of the input sequence",
    "featureCount": "Number of features per timestep"
}

# Output description
output_description = {
    "drowsiness_score": "Probability of driver drowsiness (0-1)",
    "attention_weights": "Attention weights for explainability"
}

# Convert to CoreML
coreml_model = ct.convert(
    model,
    inputs=[
        ct.TensorType(shape=(1, 600, 10), name="input_sequence")
    ],
    minimum_deployment_target=ct.target.iOS15
)

# Set metadata
coreml_model.author = "Vehicle Motion Tracking"
coreml_model.license = "Your License"
coreml_model.short_description = "Transformer-based drowsiness detection model"
coreml_model.input_description = input_description
coreml_model.output_description = output_description

# Save the model
coreml_model.save("TransformerDrowsinessModel.mlmodel")

print("Model saved as TransformerDrowsinessModel.mlmodel")
