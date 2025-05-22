# internvl/scripts/prepare_data.py

#!/usr/bin/env python3

import argparse
import sys
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_processor import DataProcessor
from transformers import AutoTokenizer

def create_sample_dataset(output_dir: Path, num_samples: int = 100):
    """Create a sample dataset for testing."""
    
    print(f"Creating sample dataset with {num_samples} examples...")
    
    # Sample questions and answers for vision-language tasks
    sample_data = []
    
    questions = [
        "What do you see in this image?",
        "Describe the scene in detail.",
        "What objects are visible in the image?",
        "Is this a safe driving situation?",
        "What should the driver be aware of?",
        "Describe the road conditions.",
        "Are there any traffic signs visible?",
        "What is the weather like in this scene?",
        "How many vehicles are in the image?",
        "What type of environment is this?"
    ]
    
    answers = [
        "I can see a road scene with various elements that are important for driving safety.",
        "This image shows a typical driving environment with road infrastructure and surroundings.",
        "The scene contains roadway elements, potential traffic, and environmental features.",
        "Based on the visual information, I can assess the driving conditions and safety factors.",
        "The driver should pay attention to road conditions, traffic, and surrounding environment.",
        "The road appears to be in normal condition for safe driving.",
        "I can identify various road signs and traffic control elements in the scene.",
        "The weather conditions appear suitable for driving with good visibility.",
        "There are several vehicles visible in the traffic scene.",
        "This appears to be an urban/suburban driving environment."
    ]
    
    import random
    
    for i in range(num_samples):
        sample = {
            "image_path": f"images/sample_{i:04d}.jpg",  # Placeholder image path
            "question": random.choice(questions),
            "answer": random.choice(answers),
            "metadata": {
                "scene_type": random.choice(["urban", "highway", "suburban", "rural"]),
                "weather": random.choice(["clear", "cloudy", "rainy", "foggy"]),
                "time_of_day": random.choice(["morning", "afternoon", "evening", "night"])
            }
        }
        sample_data.append(sample)
    
    # Split into train/val/test
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)
    
    train_data = sample_data[:train_size]
    val_data = sample_data[train_size:train_size + val_size]
    test_data = sample_data[train_size + val_size:]
    
    # Save datasets
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "train.json", 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(output_dir / "val.json", 'w') as f:
        json.dump(val_data, f, indent=2)
    
    with open(output_dir / "test.json", 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Sample dataset created:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Validation: {len(val_data)} samples") 
    print(f"  Test: {len(test_data)} samples")

def convert_csv_to_json(csv_path: Path, output_path: Path):
    """Convert CSV data to JSON format."""
    
    print(f"Converting CSV to JSON: {csv_path} -> {output_path}")
    
    df = pd.read_csv(csv_path)
    
    # Expected columns for IMU + vision data
    required_cols = ['timestamp', 'image_path', 'question', 'answer']
    optional_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
    
    if not all(col in df.columns for col in required_cols):
        print(f"Warning: Missing required columns. Expected: {required_cols}")
        print(f"Available columns: {list(df.columns)}")
    
    # Convert to JSON format
    data = []
    for _, row in df.iterrows():
        sample = {
            "image_path": row.get('image_path', ''),
            "question": row.get('question', ''),
            "answer": row.get('answer', ''),
            "timestamp": row.get('timestamp', 0)
        }
        
        # Add IMU data if available
        if all(col in df.columns for col in optional_cols):
            sample["imu_data"] = {
                "accel": [row['accel_x'], row['accel_y'], row['accel_z']],
                "gyro": [row['gyro_x'], row['gyro_y'], row['gyro_z']]
            }
        
        data.append(sample)
    
    # Save JSON
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Converted {len(data)} samples to JSON format")

def validate_dataset(data_path: Path) -> Dict:
    """Validate dataset format and content."""
    
    print(f"Validating dataset: {data_path}")
    
    if not data_path.exists():
        return {"valid": False, "error": "File does not exist"}
    
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            return {"valid": False, "error": "Data should be a list"}
        
        # Check sample format
        required_fields = ["image_path", "question", "answer"]
        sample_count = 0
        valid_samples = 0
        missing_images = 0
        
        for sample in data:
            sample_count += 1
            
            if all(field in sample for field in required_fields):
                valid_samples += 1
                
                # Check if image path exists (if it's a real path)
                image_path = Path(sample["image_path"])
                if not str(image_path).startswith("images/sample_") and not image_path.exists():
                    missing_images += 1
        
        validation_result = {
            "valid": True,
            "total_samples": sample_count,
            "valid_samples": valid_samples,
            "missing_images": missing_images,
            "validation_rate": valid_samples / sample_count if sample_count > 0 else 0
        }
        
        print(f"Validation results:")
        for key, value in validation_result.items():
            print(f"  {key}: {value}")
        
        return validation_result
        
    except Exception as e:
        return {"valid": False, "error": str(e)}

def main():
    parser = argparse.ArgumentParser(description='Prepare data for InternVL training')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing raw data')
    parser.add_argument('--output_dir', type=str, default='./data/processed',
                       help='Output directory for processed data')
    parser.add_argument('--create_sample', action='store_true',
                       help='Create sample dataset for testing')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to create (if creating sample dataset)')
    parser.add_argument('--convert_csv', type=str,
                       help='Convert CSV file to JSON format')
    parser.add_argument('--validate', action='store_true',
                       help='Validate dataset format')
    parser.add_argument('--combine_with_imu', type=str,
                       help='Path to IMU data CSV to combine with images')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    if args.create_sample:
        create_sample_dataset(output_dir, args.num_samples)
        return
    
    if args.convert_csv:
        csv_path = Path(args.convert_csv)
        output_path = output_dir / f"{csv_path.stem}.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        convert_csv_to_json(csv_path, output_path)
        return
    
    if args.validate:
        for json_file in output_dir.glob("*.json"):
            validate_dataset(json_file)
        return
    
    # Main data processing pipeline
    try:
        print("Starting data preparation pipeline...")
        
        # Initialize data processor (we'll need a tokenizer)
        print("Note: For full data processing, please run after downloading the model")
        print("This script prepares the data structure and validates formats")
        
        # Create output directories
        (output_dir / "training").mkdir(parents=True, exist_ok=True)
        (output_dir / "validation").mkdir(parents=True, exist_ok=True)
        (output_dir / "test").mkdir(parents=True, exist_ok=True)
        
        # Process IMU + vision data combination if requested
        if args.combine_with_imu:
            print(f"Combining vision data with IMU data from: {args.combine_with_imu}")
            
            # This would require the actual implementation
            # For now, create a placeholder
            imu_df = pd.read_csv(args.combine_with_imu)
            
            print(f"Loaded IMU data with {len(imu_df)} samples")
            print(f"IMU data columns: {list(imu_df.columns)}")
            
            # Here you would implement the actual combination logic
            # based on timestamps, frame numbers, etc.
            
        print("Data preparation completed!")
        print(f"Output directory: {output_dir}")
        
    except Exception as e:
        print(f"Error in data preparation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

