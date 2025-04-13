def generate_mixed_dataset(self, num_normal=100, num_accidents=50, num_drowsy=50, duration_range=(30, 120)):
    """
    Generate a complete dataset with normal, accident, and drowsy driving
    
    Args:
        num_normal: Number of normal driving sequences
        num_accidents: Number of accident sequences
        num_drowsy: Number of drowsy driving sequences
        duration_range: Min and max duration in seconds for each sequence
        
    Returns:
        DataFrame with all generated data
    """
    all_data = []
    
    print("Generating normal driving sequences...")
    for i in tqdm(range(num_normal)):
        duration = np.random.uniform(*duration_range)
        df = self.generate_normal_driving(duration)
        df['sequence_id'] = f'normal_{i}'
        all_data.append(df)
    
    print("Generating accident sequences...")
    for i in tqdm(range(num_accidents)):
        duration = np.random.uniform(*duration_range)
        accident_type = np.random.choice(['collision', 'rollover', 'spin'])
        df = self.generate_accident_events(duration, accident_type)
        df['sequence_id'] = f'accident_{i}_{accident_type}'
        df['accident_type'] = accident_type
        all_data.append(df)
    
    print("Generating drowsy driving sequences...")
    for i in tqdm(range(num_drowsy)):
        duration = np.random.uniform(*duration_range)
        drowsiness_level = np.random.choice(['mild', 'moderate', 'severe'])
        df = self.generate_drowsy_driving(duration, drowsiness_level)
        df['sequence_id'] = f'drowsy_{i}_{drowsiness_level}'
        df['drowsiness_level'] = drowsiness_level
        all_data.append(df)
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def save_dataset(self, df, filepath):
    """Save the generated dataset to CSV"""
    df.to_csv(filepath, index=False)
    print(f"Dataset saved to {filepath}")
    
def visualize_sequence(self, df, sequence_id=None, show_accidents=True, show_drowsiness=True):
    """
    Visualize a sequence from the dataset
    
    Args:
        df: DataFrame with IMU data
        sequence_id: Specific sequence to visualize (if None, choose random)
        show_accidents: Highlight accident regions
        show_drowsiness: Highlight drowsy driving regions
    """
    if sequence_id is None:
        sequence_id = np.random.choice(df['sequence_id'].unique())
    
    # Filter data for the selected sequence
    sequence_data = df[df['sequence_id'] == sequence_id]
    
    fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Plot accelerometer data
    axs[0].plot(sequence_data['time'], sequence_data['acc_x'], label='Acceleration X')
    axs[0].plot(sequence_data['time'], sequence_data['acc_y'], label='Acceleration Y')
    axs[0].plot(sequence_data['time'], sequence_data['acc_z'], label='Acceleration Z')
    axs[0].set_ylabel('Acceleration (m/s²)')
    axs[0].set_title(f'Sequence: {sequence_id}')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot gyroscope data
    axs[1].plot(sequence_data['time'], sequence_data['gyro_x'], label='Gyro X (Roll)')
    axs[1].plot(sequence_data['time'], sequence_data['gyro_y'], label='Gyro Y (Pitch)')
    axs[1].plot(sequence_data['time'], sequence_data['gyro_z'], label='Gyro Z (Yaw)')
    axs[1].set_ylabel('Angular Velocity (rad/s)')
    axs[1].set_xlabel('Time (s)')
    axs[1].legend()
    axs[1].grid(True)
    
    # Highlight accident regions
    if show_accidents and 'accident' in sequence_data['label'].values:
        accident_regions = sequence_data[sequence_data['label'] == 'accident']['time']
        for ax in axs:
            for t in accident_regions:
                ax.axvline(t, color='red', alpha=0.2)
        
        # Add a red background for accident regions
        accident_starts = []
        accident_ends = []
        in_accident = False
        
        for i, row in sequence_data.iterrows():
            if row['label'] == 'accident' and not in_accident:
                accident_starts.append(row['time'])
                in_accident = True
            elif row['label'] != 'accident' and in_accident:
                accident_ends.append(sequence_data.loc[i-1, 'time'])
                in_accident = False
        
        if in_accident:
            accident_ends.append(sequence_data['time'].iloc[-1])
        
        for start, end in zip(accident_starts, accident_ends):
            for ax in axs:
                ax.axvspan(start, end, color='red', alpha=0.2)
    
    # Highlight drowsy regions
    if show_drowsiness and 'drowsy' in sequence_data['label'].values:
        drowsy_regions = sequence_data[sequence_data['label'] == 'drowsy']['time']
        for ax in axs:
            for t in drowsy_regions:
                ax.axvline(t, color='orange', alpha=0.2)
        
        # Add an orange background for drowsy regions
        drowsy_starts = []
        drowsy_ends = []
        in_drowsy = False
        
        for i, row in sequence_data.iterrows():
            if row['label'] == 'drowsy' and not in_drowsy:
                drowsy_starts.append(row['time'])
                in_drowsy = True
            elif row['label'] != 'drowsy' and in_drowsy:
                drowsy_ends.append(sequence_data.loc[i-1, 'time'])
                in_drowsy = False
        
        if in_drowsy:
            drowsy_ends.append(sequence_data['time'].iloc[-1])
        
        for start, end in zip(drowsy_starts, drowsy_ends):
            for ax in axs:
                ax.axvspan(start, end, color='orange', alpha=0.2)
    
    plt.tight_layout()
    plt.show()
