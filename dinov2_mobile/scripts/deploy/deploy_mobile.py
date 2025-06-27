#!/usr/bin/env python3
"""
DINOv2 Mobile Deployment Script
Creates deployment packages for iOS and Android
"""
import argparse
import shutil
import zipfile
from pathlib import Path
import yaml

def create_ios_package(model_path: str, output_dir: str):
    """Create iOS deployment package."""
    ios_dir = Path(output_dir) / "iOS_Package"
    ios_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy model
    if Path(model_path).exists():
        shutil.copy2(model_path, ios_dir)
    
    # Copy Swift integration
    swift_dir = ios_dir / "Integration"
    swift_dir.mkdir(exist_ok=True)
    
    swift_source = Path("mobile/ios/swift/DINOv2Inference.swift")
    if swift_source.exists():
        shutil.copy2(swift_source, swift_dir)
    
    # Create README
    readme_content = f"""# DINOv2 iOS Deployment Package

## Files Included
- `{Path(model_path).name}`: CoreML model file
- `Integration/DINOv2Inference.swift`: Swift integration class

## Integration Steps

1. **Add Model to Xcode Project:**
   - Drag the .mlpackage file into your Xcode project
   - Ensure it's added to your target

2. **Add Swift File:**
   - Add DINOv2Inference.swift to your project

3. **Usage Example:**
   ```swift
   let inference = DINOv2Inference.shared
   
   inference.extractFeatures(from: image) {{ result in
       switch result {{
       case .success(let features):
           print("Features extracted: \\(features.count)")
       case .failure(let error):
           print("Error: \\(error)")
       }}
   }}
   ```

## Requirements
- iOS 15.0+
- Xcode 15.0+
- Device with Neural Engine (recommended)

## Performance
- Inference time: ~100-300ms
- Memory usage: ~200-400MB
- Model size: ~{Path(model_path).stat().st_size / (1024*1024):.1f}MB
"""
    
    with open(ios_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    return str(ios_dir)

def create_android_package(model_path: str, output_dir: str):
    """Create Android deployment package."""
    android_dir = Path(output_dir) / "Android_Package"
    android_dir.mkdir(parents=True, exist_ok=True)
    
    # Create assets directory
    assets_dir = android_dir / "assets"
    assets_dir.mkdir(exist_ok=True)
    
    # Copy model to assets
    if Path(model_path).exists():
        shutil.copy2(model_path, assets_dir / "dinov2_mobile.tflite")
    
    # Copy Kotlin integration
    kotlin_dir = android_dir / "kotlin"
    kotlin_dir.mkdir(exist_ok=True)
    
    kotlin_source = Path("mobile/android/kotlin/DINOv2Inference.kt")
    if kotlin_source.exists():
        shutil.copy2(kotlin_source, kotlin_dir)
    
    # Create gradle dependencies
    gradle_content = """dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
}"""
    
    with open(android_dir / "build.gradle.dependencies", "w") as f:
        f.write(gradle_content)
    
    # Create README
    readme_content = f"""# DINOv2 Android Deployment Package

## Files Included
- `assets/dinov2_mobile.tflite`: TensorFlow Lite model
- `kotlin/DINOv2Inference.kt`: Kotlin integration class
- `build.gradle.dependencies`: Required dependencies

## Integration Steps

1. **Add Dependencies:**
   Add the contents of `build.gradle.dependencies` to your app's build.gradle

2. **Add Model Asset:**
   Copy `dinov2_mobile.tflite` to your app's `assets` folder

3. **Add Kotlin Class:**
   Add `DINOv2Inference.kt` to your project

4. **Usage Example:**
   ```kotlin
   val inference = DINOv2Inference(context)
   inference.initialize()
   
   val features = inference.extractFeatures(bitmap)
   features?.let {{
       println("Features extracted: ${{it.size}}")
   }}
   ```

## Requirements
- Android API 24+ (Android 7.0+)
- 4GB+ RAM recommended
- GPU acceleration supported

## Performance
- Inference time: ~200-500ms
- Memory usage: ~300-600MB
- Model size: ~{Path(model_path).stat().st_size / (1024*1024):.1f}MB
"""
    
    with open(android_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    return str(android_dir)

def main():
    parser = argparse.ArgumentParser(description="Create DINOv2 mobile deployment packages")
    parser.add_argument("--models-dir", default="./models/converted", help="Converted models directory")
    parser.add_argument("--output", default="./deployment_packages", help="Output directory")
    parser.add_argument("--platforms", nargs="+", choices=["ios", "android"], 
                       default=["ios", "android"], help="Target platforms")
    parser.add_argument("--zip", action="store_true", help="Create zip archives")
    
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    packages = {}
    
    for platform in args.platforms:
        print(f"📦 Creating {platform.upper()} package...")
        
        if platform == "ios":
            model_file = models_dir / "dinov2_mobile.mlpackage"
            if model_file.exists():
                package_dir = create_ios_package(str(model_file), str(output_dir))
                packages[platform] = package_dir
                print(f"✅ iOS package created: {package_dir}")
            else:
                print(f"❌ iOS model not found: {model_file}")
                
        elif platform == "android":
            model_file = models_dir / "dinov2_mobile.tflite"
            if model_file.exists():
                package_dir = create_android_package(str(model_file), str(output_dir))
                packages[platform] = package_dir
                print(f"✅ Android package created: {package_dir}")
            else:
                print(f"❌ Android model not found: {model_file}")
    
    # Create zip archives if requested
    if args.zip:
        print("\n📦 Creating zip archives...")
        for platform, package_dir in packages.items():
            zip_path = f"{package_dir}.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in Path(package_dir).rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(package_dir)
                        zipf.write(file_path, arcname)
            print(f"📦 {platform.upper()} archive: {zip_path}")
    
    print("\n🎉 Deployment packages created successfully!")

if __name__ == "__main__":
    main()
