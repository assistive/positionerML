# internvl/scripts/deploy.py

#!/usr/bin/env python3

import argparse
import sys
import shutil
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from deployment_utils import DeploymentUtils

def main():
    parser = argparse.ArgumentParser(description='Deploy InternVL models to target platforms')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing converted mobile models')
    parser.add_argument('--platform', type=str, choices=['ios', 'android'], required=True,
                       help='Target deployment platform')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for deployment package')
    parser.add_argument('--include_examples', action='store_true',
                       help='Include example code and integration samples')
    parser.add_argument('--create_package', action='store_true',
                       help='Create deployment package with all necessary files')
    
    args = parser.parse_args()
    
    try:
        print(f"Preparing {args.platform} deployment package...")
        
        # Initialize deployment utils
        deployment = DeploymentUtils()
        
        # Create deployment package
        package_path = deployment.create_deployment_package(
            model_dir=args.model_dir,
            platform=args.platform,
            output_dir=args.output_dir,
            include_examples=args.include_examples
        )
        
        print(f"Deployment package created at: {package_path}")
        
        if args.create_package:
            # Create compressed package
            archive_path = deployment.create_archive(package_path)
            print(f"Compressed package: {archive_path}")
        
        print("Deployment preparation completed successfully!")
        
    except Exception as e:
        print(f"Error during deployment preparation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

