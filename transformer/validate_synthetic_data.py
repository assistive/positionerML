def validate_synthetic_data(self, real_data_path=None):
    """
    Validate that synthetic data has similar statistical properties to real data
    
    Args:
        real_data_path: Path to real IMU data for comparison (optional)
    """
    # Generate a sample of synthetic data
    synthetic_df = self.generate_mixed_dataset(num_normal=20, num_accidents=10, num_drowsy=10)
    
    # Calculate statistical properties of synthetic data
    syn_stats = {
        'acc_x_mean': synthetic_df['acc_x'].mean(),
        'acc_x_std': synthetic_df['acc_x'].std(),
        'acc_y_mean': synthetic_df['acc_y'].mean(),
        'acc_y_std': synthetic_df['acc_y'].std(),
        'acc_z_mean': synthetic_df['acc_z'].mean(),
        'acc_z_std': synthetic_df['acc_z'].std(),
        'gyro_x_mean': synthetic_df['gyro_x'].mean(),
        'gyro_x_std': synthetic_df['gyro_x'].std(),
        'gyro_y_mean': synthetic_df['gyro_y'].mean(),
        'gyro_y_std': synthetic_df['gyro_y'].std(),
        'gyro_z_mean': synthetic_df['gyro_z'].mean(),
        'gyro_z_std': synthetic_df['gyro_z'].std(),
    }
    
    # Calculate frequency content
    syn_fft_acc_x = np.abs(np.fft.fft(synthetic_df['acc_x'].values))
    syn_fft_acc_y = np.abs(np.fft.fft(synthetic_df['acc_y'].values))
    syn_fft_acc_z = np.abs(np.fft.fft(synthetic_df['acc_z'].values))
    syn_fft_gyro_z = np.abs(np.fft.fft(synthetic_df['gyro_z'].values))
    
    print("Synthetic Data Statistics:")
    for key, value in syn_stats.items():
        print(f"  {key}: {value:.4f}")
    
    # If real data is provided, compare statistics
    if real_data_path and os.path.exists(real_data_path):
        real_df = pd.read_csv(real_data_path)
        
        # Calculate statistical properties of real data
        real_stats = {
            'acc_x_mean': real_df['acc_x'].mean(),
            'acc_x_std': real_df['acc_x'].std(),
            'acc_y_mean': real_df['acc_y'].mean(),
            'acc_y_std': real_df['acc_y'].std(),
            'acc_z_mean': real_df['acc_z'].mean(),
            'acc_z_std': real_df['acc_z'].std(),
            'gyro_x_mean': real_df['gyro_x'].mean(),
            'gyro_x_std': real_df['gyro_x'].std(),
            'gyro_y_mean': real_df['gyro_y'].mean(),
            'gyro_y_std': real_df['gyro_y'].std(),
            'gyro_z_mean': real_df['gyro_z'].mean(),
            'gyro_z_std': real_df['gyro_z'].std(),
        }
        
        # Calculate frequency content
        real_fft_acc_x = np.abs(np.fft.fft(real_df['acc_x'].values))
        real_fft_acc_y = np.abs(np.fft.fft(real_df['acc_y'].values))
        real_fft_acc_z = np.abs(np.fft.fft(real_df['acc_z'].values))
        real_fft_gyro_z = np.abs(np.fft.fft(real_df['gyro_z'].values))
        
        print("\nReal Data Statistics:")
        for key, value in real_stats.items():
            print(f"  {key}: {value:.4f}")
        
        # Plot comparison of real vs synthetic
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        
        # Acceleration histograms
        axs[0, 0].hist(synthetic_df['acc_x'], bins=50, alpha=0.5, label='Synthetic')
        axs[0, 0].hist(real_df['acc_x'], bins=50, alpha=0.5, label='Real')
        axs[0, 0].set_title('Acceleration X Distribution')
        axs[0, 0].legend()
        
        axs[0, 1].hist(synthetic_df['acc_y'], bins=50, alpha=0.5, label='Synthetic')
        axs[0, 1].hist(real_df['acc_y'], bins=50, alpha=0.5, label='Real')
        axs[0, 1].set_title('Acceleration Y Distribution')
        
        axs[0, 2].hist(synthetic_df['acc_z'], bins=50, alpha=0.5, label='Synthetic')
        axs[0, 2].hist(real_df['acc_z'], bins=50, alpha=0.5, label='Real')
        axs[0, 2].set_title('Acceleration Z Distribution')
        
        # Gyroscope histograms
        axs[1, 0].hist(synthetic_df['gyro_x'], bins=50, alpha=0.5, label='Synthetic')
        axs[1, 0].hist(real_df['gyro_x'], bins=50, alpha=0.5, label='Real')
        axs[1, 0].set_title('Gyro X Distribution')
        
        axs[1, 1].hist(synthetic_df['gyro_y'], bins=50, alpha=0.5, label='Synthetic')
        axs[1, 1].hist(real_df['gyro_y'], bins=50, alpha=0.5, label='Real')
        axs[1, 1].set_title('Gyro Y Distribution')
        
        axs[1, 2].hist(synthetic_df['gyro_z'], bins=50, alpha=0.5, label='Synthetic')
        axs[1, 2].hist(real_df['gyro_z'], bins=50, alpha=0.5, label='Real')
        axs[1, 2].set_title('Gyro Z Distribution')
        
        plt.tight_layout()
        plt.show()
    
    # Visualize frequency content
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(syn_fft_acc_x[:len(syn_fft_acc_x)//10])  # Plot first 10% of frequencies
    plt.title('Synthetic Acceleration X Frequency Content')
    plt.xlabel('Frequency Bin')
    plt.ylabel('Magnitude')
    
    plt.subplot(2, 2, 2)
    plt.plot(syn_fft_acc_z[:len(syn_fft_acc_z)//10])
    plt.title('Synthetic Acceleration Z Frequency Content')
    plt.xlabel('Frequency Bin')
    
    plt.subplot(2, 2, 3)
    plt.plot(syn_fft_gyro_z[:len(syn_fft_gyro_z)//10])
    plt.title('Synthetic Gyro Z Frequency Content')
    plt.xlabel('Frequency Bin')
    plt.ylabel('Magnitude')
    
    plt.tight_layout()
    plt.show()
