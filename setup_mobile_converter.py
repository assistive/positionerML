#!/usr/bin/env python3
"""
Setup script for the Unified VLM Mobile Converter
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_system_requirements():
    """Check if system meets requirements."""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {python_version.major}.{python_version.minor}")
    
    # Check available memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 16:
            print(f"⚠️  Low memory: {memory_gb:.1f}GB (16GB+ recommended)")
        else:
            print(f"✅ Memory: {memory_gb:.1f}GB")
    except ImportError:
        print("⚠️  Cannot check memory (psutil not installed)")
    
    # Check disk space
    disk_free = os.statvfs('.').f_bavail * os.statvfs('.').f_frsize / (1024**3)
    if disk_free < 100:
        print(f"⚠️  Low disk space: {disk_free:.1f}GB (100GB+ recommended)")
    else:
        print(f"✅ Disk space: {disk_free:.1f}GB available")
    
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    # Core dependencies
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-r", "requirements_mobile.txt"
    ], check=True)
    
    # Platform-specific dependencies
    if platform.system() == "Darwin":  # macOS
        print("🍎 Installing iOS development tools...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "coremltools"
        ])
    
    print("✅ Dependencies installed successfully")

def setup_directories():
    """Create necessary directories."""
    print("\n📁 Setting up directories...")
    
    directories = [
        "mobile_models",
        "mobile_models/ios", 
        "mobile_models/android",
        "model_cache",
        "conversion_reports",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  Created: {directory}")

def main():
    """Main setup function."""
    print("🚀 Setting up Unified VLM Mobile Converter\n")
    
    try:
        if not check_system_requirements():
            print("❌ System requirements not met")
            return 1
        
        install_dependencies()
        setup_directories()
        
        print("\n✅ Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Place your VLM models in the respective directories:")
        print("   - fastvlm/models/pretrained/")
        print("   - internvl/models/pretrained/")
        print("   - qwen-vl-service/models/pretrained/")
        print("\n2. Run the converter:")
        print("   python unified_mobile_converter.py --discover-only")
        print("   python unified_mobile_converter.py --all")
        print("\n3. Check results in mobile_models/ directory")
        
        return 0
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
