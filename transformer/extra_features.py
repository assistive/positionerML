def extract_features(imu_data):
    """
    Extract engineered features from raw IMU data
    
    Args:
        imu_data: Raw IMU readings [n_samples, 6] (accel_x/y/z, gyro_x/y/z)
        
    Returns:
        Feature matrix [n_samples, 8+] with engineered features
    """
    # Calculate magnitudes
    acc_mag = np.linalg.norm(imu_data[:, 0:3], axis=1)
    gyro_mag = np.linalg.norm(imu_data[:, 3:6], axis=1)
    
    # Calculate jerk (derivative of acceleration)
    acc_jerk = np.gradient(imu_data[:, 0:3], axis=0)
    acc_jerk_mag = np.linalg.norm(acc_jerk, axis=1)
    
    # Calculate angular acceleration (derivative of gyro)
    angular_acc = np.gradient(imu_data[:, 3:6], axis=0)
    angular_acc_mag = np.linalg.norm(angular_acc, axis=1)
    
    # Create feature matrix
    features = np.column_stack([
        imu_data,                  # Original 6 IMU values
        acc_mag, gyro_mag,         # Magnitude values
        acc_jerk_mag,              # Jerk for detecting sudden changes
        angular_acc_mag,           # Angular acceleration
        # Add more engineered features as needed
    ])
    
    return features
