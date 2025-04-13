generator = IMUDataGenerator()
dataset = generator.generate_mixed_dataset()
generator.save_dataset(dataset, 'synthetic_training_data.csv')
