def generate_synthetic_dataset():
    """Generate a complete synthetic dataset for training and testing"""
    generator = IMUDataGenerator(sampling_rate=100, random_seed=42)
    
    # Generate a balanced dataset
    dataset = generator.generate_mixed_dataset(
        num_normal=200,      # 200 normal driving sequences
        num_accidents=100,   # 100 accident sequences
        num_drowsy=100,      # 100 drowsy driving sequences
        duration_range=(30, 120)  # Each sequence 30-120 seconds
    )
    
    # Save the dataset
    generator.save_dataset(dataset, 'synthetic_imu_dataset.csv')
    
    # Visualize a few examples
    print("\nVisualizing examples:")
    for _ in range(3):
        generator.visualize_sequence(dataset)
    
    # Show specific accident and drowsy examples
    accident_id = dataset[dataset['label'] == 'accident']['sequence_id'].unique()[0]
    print(f"\nVisuralizing accident example: {accident_id}")
    generator.visualize_sequence(dataset, sequence_id=accident_id)
    
    drowsy_id = dataset[dataset['label'] == 'drowsy']['sequence_id'].unique()[0]
    print(f"\nVisuralizing drowsy driving example: {drowsy_id}")
    generator.visualize_sequence(dataset, sequence_id=drowsy_id)
    
    return dataset
