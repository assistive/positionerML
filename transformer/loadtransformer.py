class TransformerBasedDetector:
    def __init__(self, model_path_accident, model_path_drowsiness, window_size=200, device=None):
        # Set device
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = IMUTransformer()
        
        # Load task-specific model weights as needed
        self.accident_model_path = model_path_accident
        self.drowsiness_model_path = model_path_drowsiness
        
        # Current task and model
        self.current_task = None
        
        # Buffer for storing incoming IMU data
        self.window_size = window_size
        self.imu_buffer = []
        
        # Load normalization params
        self.scaler = StandardScaler()  # This would be trained and saved during model training
        
    def _load_model(self, task):
        """Load the appropriate model weights for the specified task"""
        if task == 'accident':
            self.model.load_state_dict(torch.load(self.accident_model_path, map_location=self.device))
        elif task == 'drowsiness':
            self.model.load_state_dict(torch.load(self.drowsiness_model_path, map_location=self.device))
        else:
            raise ValueError(f"Unknown task: {task}")
            
        self.current_task = task
        self.model.to(self.device)
        self.model.eval()
        
    def process_imu(self, imu_reading, task='accident'):
        """
        Process a new IMU reading and return detection results
        
        Args:
            imu_reading: New IMU reading (accel_x/y/z, gyro_x/y/z)
            task: Detection task to perform
            
        Returns:
            Detection result and confidence score
        """
        # Check if we need to load a different model
        if self.current_task != task:
            self._load_model(task)
            
        # Add reading to buffer
        self.imu_buffer.append(imu_reading)
        
        # Keep buffer at window size
        if len(self.imu_buffer) > self.window_size:
            self.imu_buffer.pop(0)
            
        # Only make predictions when we have a full window
        if len(self.imu_buffer) < self.window_size:
            return None, 0.0
        
        # Preprocess data
        features = extract_features(np.array(self.imu_buffer))
        features = self.scaler.transform(features)
        
        # Convert to tensor
        tensor_features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(tensor_features, task=task)
            probabilities = F.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, prediction].item()
            
        return prediction, confidence
