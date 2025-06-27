#!/usr/bin/env python3
"""
Download DINOv2 models from torch.hub
"""
import torch
import argparse
from pathlib import Path

def download_model(model_name: str, output_dir: str):
    """Download DINOv2 model."""
    print(f"📥 Downloading {model_name}...")
    
    # Load model from torch.hub
    model = torch.hub.load('facebookresearch/dinov2', model_name)
    
    # Save model
    output_path = Path(output_dir) / f"{model_name}.pth"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), output_path)
    print(f"✅ Model saved to: {output_path}")
    
    # Save full model for mobile conversion
    full_model_path = Path(output_dir) / f"{model_name}_full.pth"
    torch.save(model, full_model_path)
    print(f"✅ Full model saved to: {full_model_path}")
    
    return str(output_path)

def main():
    parser = argparse.ArgumentParser(description="Download DINOv2 models")
    parser.add_argument("--models", nargs="+", 
                       choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"],
                       default=["dinov2_vits14"],
                       help="Models to download")
    parser.add_argument("--output", default="./models/pretrained", help="Output directory")
    
    args = parser.parse_args()
    
    for model_name in args.models:
        try:
            download_model(model_name, args.output)
        except Exception as e:
            print(f"❌ Failed to download {model_name}: {e}")

if __name__ == "__main__":
    main()
