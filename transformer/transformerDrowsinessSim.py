def generate_drowsy_driving(self, duration_seconds, drowsiness_level='moderate'):
    """
    Generate synthetic IMU data with drowsy driving patterns
    
    Args:
        duration_seconds: Length of data to generate in seconds
        drowsiness_level: Level of drowsiness ('mild', 'moderate', 'severe')
        
    Returns:
        DataFrame with synthetic IMU data showing drowsy driving
    """
    # Generate normal driving as baseline
    df = self.generate_normal_driving(duration_seconds)
    
    # Set normal driving label
    df['label'] = 'normal'
    
    # Parameters based on drowsiness level
    if drowsiness_level == 'mild':
        lane_drift_freq = 0.05  # One drift every ~20 seconds
        correction_magnitude = 0.2  # Smaller corrections
        delay_factor = 1.2  # Slightly delayed reactions
    elif drowsiness_level == 'moderate':
        lane_drift_freq = 0.1   # One drift every ~10 seconds
        correction_magnitude = 0.5  # Medium corrections
        delay_factor = 1.5  # More delayed reactions
    else:  # severe
        lane_drift_freq = 0.15  # One drift every ~6-7 seconds
        correction_magnitude = 0.8  # Larger corrections
        delay_factor = 2.0  # Very delayed reactions
    
    # Number of lane drift events
    num_drifts = int(duration_seconds * lane_drift_freq)
    
    # Add drowsy driving events
    drowsy_indices = []
    
    for _ in range(num_drifts):
        # Choose random start point (avoid the first and last 5 seconds)
        start_idx = np.random.randint(5 * self.sampling_rate, 
                                       (duration_seconds - 5) * self.sampling_rate)
        
        # Drift duration: longer with more severe drowsiness
        drift_duration = int(self.sampling_rate * (2 + 2 * delay_factor * np.random.random()))
        end_idx = min(start_idx + drift_duration, len(df) - 1)
        
        # Record indices for labeling
        event_indices = list(range(start_idx, end_idx))
        drowsy_indices.extend(event_indices)
        
        # Create drift pattern: slow drift followed by sudden correction
        drift_portion = int(0.8 * drift_duration)  # 80% of time is slow drift
        correction_portion = drift_duration - drift_portion  # 20% is correction
        
        # Slow drift in steering angle (gyro_z)
        drift_direction = 1 if np.random.random() > 0.5 else -1
        drift_curve = np.linspace(0, drift_direction * 0.05, drift_portion)
        df.loc[start_idx:start_idx+drift_portion-1, 'gyro_z'] += drift_curve
        
        # Sudden correction
        correction = -drift_direction * correction_magnitude
        correction_curve = np.ones(correction_portion) * correction
        df.loc[start_idx+drift_portion:end_idx-1, 'gyro_z'] += correction_curve
        
        # Add corresponding lateral acceleration
        df.loc[start_idx:start_idx+drift_portion-1, 'acc_y'] += drift_curve * 0.5
        df.loc[start_idx+drift_portion:end_idx-1, 'acc_y'] += correction_curve * 2.0
    
    # Label drowsy events
    df.loc[drowsy_indices, 'label'] = 'drowsy'
    
    # Add micro-corrections - characteristic of drowsy driving
    if drowsiness_level in ['moderate', 'severe']:
        num_micro = int(duration_seconds / 4)  # One every 4 seconds on average
        
        for _ in range(num_micro):
            # Random position
            pos = np.random.randint(self.sampling_rate, len(df) - self.sampling_rate)
            
            # Very brief correction (0.2-0.4s)
            duration = int(self.sampling_rate * (0.2 + 0.2 * np.random.random()))
            micro_amp = 0.1 + 0.2 * np.random.random()
            direction = 1 if np.random.random() > 0.5 else -1
            
            # Apply micro-correction
            micro_curve = signal.gaussian(duration, std=duration/5) * micro_amp * direction
            df.loc[pos:pos+duration-1, 'gyro_z'] += micro_curve
            
            # Mark these as drowsy too
            df.loc[pos:pos+duration-1, 'label'] = 'drowsy'
    
    return df
