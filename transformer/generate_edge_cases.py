def generate_edge_cases(self, num_cases=10):
    """
    Generate challenging edge cases for model training
    
    Returns:
        DataFrame with edge case IMU data
    """
    edge_cases = []
    
    # 1. Near-accidents (sudden braking but not crash)
    for i in range(num_cases):
        df = self.generate_normal_driving(30)  # 30s sequence
        
        # Add sudden braking event
        event_idx = np.random.randint(10 * self.sampling_rate, 20 * self.sampling_rate)
        event_duration = int(1.5 * self.sampling_rate)  # 1.5s event
        
        # Create sharp braking profile
        braking_magnitude = 5.0 + 4.0 * np.random.random()  # 5-9 m/s²
        braking_curve = signal.gaussian(event_duration, std=event_duration/6) * braking_magnitude * -1
        
        # Apply braking
        df.loc[event_idx:event_idx+event_duration-1, 'acc_x'] += braking_curve
        
        # Label this as "near_accident"
        df['label'] = 'normal'
        df.loc[event_idx:event_idx+event_duration-1, 'label'] = 'near_accident'
        df['sequence_id'] = f'near_accident_{i}'
        
        edge_cases.append(df)
    
    # 2. Vibration from rough road (could be confused with accident)
    for i in range(num_cases):
        df = self.generate_normal_driving(30)
        
        # Add high-frequency vibration
        event_idx = np.random.randint(10 * self.sampling_rate, 20 * self.sampling_rate)
        event_duration = int(5 * self.sampling_rate)  # 5s event
        
        # Create vibration
        t = np.linspace(0, event_duration/self.sampling_rate, event_duration)
        vib_freq = 20 + 10 * np.random.random()  # 20-30 Hz
        vib_amp = 1.0 + 1.0 * np.random.random()  # 1-2 m/s²
        
        vib_z = vib_amp * np.sin(2 * np.pi * vib_freq * t)
        vib_window = signal.windows.tukey(len(vib_z), alpha=0.25)
        vib_z = vib_z * vib_window
        
        # Apply vibration
        df.loc[event_idx:event_idx+event_duration-1, 'acc_z'] += vib_z
        
        # Label as "rough_road"
        df['label'] = 'normal'
        df.loc[event_idx:event_idx+event_duration-1, 'label'] = 'rough_road'
        df['sequence_id'] = f'rough_road_{i}'
        
        edge_cases.append(df)
    
    # 3. Fatigued but not drowsy (subtle indicators)
    for i in range(num_cases):
        df = self.generate_normal_driving(30)
        
        # Add occasional small corrections
        num_corrections = np.random.randint(8, 15)
        for j in range(num_corrections):
            corr_idx = np.random.randint(self.sampling_rate, len(df) - self.sampling_rate)
            corr_duration = int(0.3 * self.sampling_rate)  # 300ms
            
            # Small correction
            corr_mag = 0.1 + 0.1 * np.random.random()  # 0.1-0.2 rad/s
            direction = 1 if np.random.random() > 0.5 else -1
            
            corr_curve = signal.gaussian(corr_duration, std=corr_duration/5) * corr_mag * direction
            df.loc[corr_idx:corr_idx+corr_duration-1, 'gyro_z'] += corr_curve
            
            # Label as "fatigue"
            df.loc[corr_idx:corr_idx+corr_duration-1, 'label'] = 'fatigue'
        
        df['label'].fillna('normal', inplace=True)
        df['sequence_id'] = f'fatigue_{i}'
        
        edge_cases.append(df)
    
    # Combine all edge cases
    combined_df = pd.concat(edge_cases, ignore_index=True)
    return combined_df
