def generate_accident_events(self, duration_seconds, accident_type='collision'):
    """
    Generate synthetic IMU data with accident events
    
    Args:
        duration_seconds: Length of data to generate in seconds
        accident_type: Type of accident ('collision', 'rollover', 'spin')
        
    Returns:
        DataFrame with synthetic IMU data including accident
    """
    # Generate normal driving as the baseline
    df = self.generate_normal_driving(duration_seconds)
    
    # Choose random accident time (at least 3 seconds after start and 3 seconds before end)
    min_time = 3
    max_time = duration_seconds - 3
    accident_time = min_time + (max_time - min_time) * np.random.random()
    
    # Get the index corresponding to accident time
    accident_idx = int(accident_time * self.sampling_rate)
    
    # Set accident window (typically 1-2 seconds for impact)
    accident_duration = 0.5 + 1.5 * np.random.random()  # 0.5-2 seconds
    end_idx = min(accident_idx + int(accident_duration * self.sampling_rate), len(df))
    
    # Create impact window
    impact_indices = range(accident_idx, end_idx)
    
    # Set normal driving label
    df['label'] = 'normal'
    
    if accident_type == 'collision':
        # Front/rear collision
        # Huge spike in longitudinal acceleration
        collision_magnitude = 10.0 + 20.0 * np.random.random()  # 10-30 m/s^2
        direction = 1 if np.random.random() > 0.5 else -1  # Front or rear
        
        # Create very sharp impulse for collision
        impact_duration = int(0.3 * self.sampling_rate)  # 300ms impact
        impact_curve = signal.gaussian(impact_duration, std=impact_duration/6)
        impact_curve = impact_curve / np.max(impact_curve) * collision_magnitude * direction
        
        # Apply impact to longitudinal acceleration
        df.loc[accident_idx:accident_idx+impact_duration-1, 'acc_x'] += impact_curve
        
        # Add secondary oscillations after impact
        oscillation_duration = int(1.5 * self.sampling_rate)  # 1.5s of oscillation
        if accident_idx + impact_duration + oscillation_duration < len(df):
            oscillation_curve = np.exp(-np.linspace(0, 4, oscillation_duration)) * np.sin(np.linspace(0, 10*np.pi, oscillation_duration))
            oscillation_curve = oscillation_curve * collision_magnitude * 0.3 * direction
            df.loc[accident_idx+impact_duration:accident_idx+impact_duration+oscillation_duration-1, 'acc_x'] += oscillation_curve
        
    elif accident_type == 'rollover':
        # Vehicle rollover - dramatic changes in roll (gyro_x)
        rollover_magnitude = np.pi + np.pi/2 * np.random.random()  # 180-270 degrees/s
        rollover_duration = int(2.0 * self.sampling_rate)  # 2s rollover
        
        # Create rollover profile
        t_rollover = np.linspace(0, 1, rollover_duration)
        rollover_curve = rollover_magnitude * np.sin(np.pi * t_rollover)
        
        # Apply to gyro_x (roll)
        if accident_idx + rollover_duration < len(df):
            df.loc[accident_idx:accident_idx+rollover_duration-1, 'gyro_x'] += rollover_curve
            
        # Also add vertical and lateral acceleration components
        df.loc[accident_idx:accident_idx+rollover_duration-1, 'acc_y'] += 5.0 * np.sin(2*np.pi * t_rollover)
        df.loc[accident_idx:accident_idx+rollover_duration-1, 'acc_z'] += -9.81 * np.cos(np.pi * t_rollover) + 9.81
        
    elif accident_type == 'spin':
        # Vehicle spin - high yaw rate (gyro_z)
        spin_magnitude = np.pi/2 + np.pi/2 * np.random.random()  # 90-180 degrees/s
        spin_duration = int(3.0 * self.sampling_rate)  # 3s spin
        
        # Create spin profile with initial spike then decay
        t_spin = np.linspace(0, 1, spin_duration)
        spin_curve = spin_magnitude * np.exp(-2*t_spin) * (1 - np.cos(2*np.pi * t_spin))
        
        # Apply to gyro_z (yaw)
        if accident_idx + spin_duration < len(df):
            df.loc[accident_idx:accident_idx+spin_duration-1, 'gyro_z'] += spin_curve
            
        # Also add lateral acceleration (centripetal)
        df.loc[accident_idx:accident_idx+spin_duration-1, 'acc_y'] += 3.0 * np.sin(4*np.pi * t_spin)
    
    # Label the accident period
    df.loc[impact_indices, 'label'] = 'accident'
    
    return df
