def main():
    """Generate and validate synthetic IMU dataset for accident and drowsiness detection"""
    print("Initializing IMU Data Generator...")
    generator = IMUDataGenerator(sampling_rate=100)
    
    print("\n=== Generating Complete Synthetic Dataset ===")
    # Generate balanced dataset with various driving conditions
    dataset = generator.generate_mixed_dataset(
        num_normal=200,
        num_accidents=100,
        num_drowsy=100,
        duration_range=(30, 120)
    )
    
    print("\n=== Generating Edge Cases ===")
    edge_cases = generator.generate_edge_cases(num_cases=50)
    
    # Combine main dataset with edge cases
    full_dataset = pd.concat([dataset, edge_cases], ignore_index=True)
    
    print("\n=== Dataset Statistics ===")
    print(f"Total sequences: {full_dataset['sequence_id'].nunique()}")
    print(f"Total samples: {len(full_dataset)}")
    
    # Count each label type
    label_counts = full_dataset['label'].value_counts()
    print("\nLabel distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} samples ({count/len(full_dataset)*100:.2f}%)")
    
    # Save the dataset
    generator.save_dataset(full_dataset, 'full_synthetic_imu_dataset.csv')
    
    # Visualize examples
    print("\n=== Visualizing Examples ===")
    
    print("\n1. Normal driving example:")
    normal_id = np.random.choice(full_dataset[full_dataset['label'] == 'normal']['sequence_id'].unique())
    generator.visualize_sequence(full_dataset, sequence_id=normal_id)
    
    print("\n2. Accident example:")
    accident_id = np.random.choice(full_dataset[full_dataset['label'] == 'accident']['sequence_id'].unique())
    generator.visualize_sequence(full_dataset, sequence_id=accident_id)
    
    print("\n3. Drowsy driving example:")
    drowsy_id = np.random.choice(full_dataset[full_dataset['label'] == 'drowsy']['sequence_id'].unique())
    generator.visualize_sequence(full_dataset, sequence_id=drowsy_id)
    
    print("\n4. Edge case example:")
    edge_id = np.random.choice(edge_cases['sequence_id'].unique())
    generator.visualize_sequence(full_dataset, sequence_id=edge_id)
    
    print("\n=== Dataset Generation Complete ===")
    print(f"Full dataset saved to 'full_synthetic_imu_dataset.csv'")
    print("You can now use this dataset to train your transformer models.")

if __name__ == "__main__":
    main()
