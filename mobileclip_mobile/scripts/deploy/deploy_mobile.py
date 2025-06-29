#!/usr/bin/env python3
"""
MobileCLIP Mobile Deployment Script
Creates deployment packages for iOS and Android
"""
import argparse
import shutil
import zipfile
import json
from pathlib import Path
import yaml
from datetime import datetime

def create_ios_package(model_name: str, model_path: str, output_dir: str):
    """Create iOS deployment package."""
    ios_dir = Path(output_dir) / "iOS_Package"
    ios_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy CoreML models
    if Path(model_path).exists():
        if Path(model_path).is_dir():  # CoreML package
            shutil.copytree(model_path, ios_dir / Path(model_path).name, dirs_exist_ok=True)
        else:
            shutil.copy2(model_path, ios_dir)
    
    # Copy Swift integration
    swift_dir = ios_dir / "Integration"
    swift_dir.mkdir(exist_ok=True)
    
    swift_source = Path("mobile/ios/swift/MobileCLIPInference.swift")
    if swift_source.exists():
        shutil.copy2(swift_source, swift_dir)
    
    # Create info file
    info = {
        "model_name": model_name,
        "platform": "ios",
        "created_at": datetime.now().isoformat(),
        "requirements": {
            "ios_version": "15.0+",
            "xcode_version": "15.0+",
            "frameworks": ["CoreML", "Vision", "Accelerate"]
        }
    }
    
    with open(ios_dir / "deployment_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    # Create README
    readme_content = f"""# MobileCLIP iOS Deployment Package

## Model: {model_name}

### Files Included
- `{Path(model_path).name}`: CoreML model package
- `Integration/MobileCLIPInference.swift`: Swift integration class
- `deployment_info.json`: Deployment information

### Integration Steps

1. **Add Model to Xcode Project:**
   - Drag the `.mlpackage` files into your Xcode project
   - Ensure they're added to your target

2. **Add Swift File:**
   - Add `MobileCLIPInference.swift` to your project

3. **Usage Example:**
   ```swift
   let inference = MobileCLIPInference.shared
   
   let labels = ["dog", "cat", "bird"]
   inference.zeroShotClassify(image: image, labels: labels) {{ result in
       switch result {{
       case .success(let results):
           for (label, confidence) in results.prefix(3) {{
               print("\\(label): \\(confidence)")
           }}
       case .failure(let error):
           print("Error: \\(error)")
       }}
   }}
   ```

### Requirements
- iOS 15.0+
- Xcode 15.0+
- Device with Neural Engine (recommended)

### Performance
- Inference time: ~30-200ms (device dependent)
- Memory usage: ~150-350MB
- Optimized for Neural Engine acceleration

### Support
For issues and questions, refer to the main documentation.
"""
    
    with open(ios_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    return str(ios_dir)

def create_android_package(model_name: str, model_path: str, output_dir: str):
    """Create Android deployment package."""
    android_dir = Path(output_dir) / "Android_Package"
    android_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy TensorFlow Lite models
    if Path(model_path).exists():
        shutil.copy2(model_path, android_dir)
    
    # Copy Kotlin integration
    kotlin_dir = android_dir / "Integration"
    kotlin_dir.mkdir(exist_ok=True)
    
    kotlin_source = Path("mobile/android/kotlin/MobileCLIPInference.kt")
    if kotlin_source.exists():
        shutil.copy2(kotlin_source, kotlin_dir)
    
    # Create build.gradle dependencies
    gradle_deps = """
// Add to your app-level build.gradle dependencies block
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.6.4'
}
"""
    
    with open(android_dir / "build.gradle.dependencies", 'w') as f:
        f.write(gradle_deps)
    
    # Create info file
    info = {
        "model_name": model_name,
        "platform": "android",
        "created_at": datetime.now().isoformat(),
        "requirements": {
            "android_api": "24+",
            "compile_sdk": "34",
            "dependencies": [
                "tensorflow-lite:2.13.0",
                "tensorflow-lite-gpu:2.13.0",
                "tensorflow-lite-support:0.4.4"
            ]
        }
    }
    
    with open(android_dir / "deployment_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    # Create README
    readme_content = f"""# MobileCLIP Android Deployment Package

## Model: {model_name}

### Files Included
- `{Path(model_path).name}`: TensorFlow Lite model file
- `Integration/MobileCLIPInference.kt`: Kotlin integration class
- `build.gradle.dependencies`: Required dependencies
- `deployment_info.json`: Deployment information

### Integration Steps

1. **Add Model to Assets:**
   - Copy `.tflite` files to `app/src/main/assets/`

2. **Add Dependencies:**
   - Add contents of `build.gradle.dependencies` to your app's build.gradle

3. **Add Kotlin Class:**
   - Add `MobileCLIPInference.kt` to your project

4. **Usage Example:**
   ```kotlin
   class MainActivity : AppCompatActivity() {{
       private lateinit var mobileCLIP: MobileCLIPInference
       
       override fun onCreate(savedInstanceState: Bundle?) {{
           super.onCreate(savedInstanceState)
           
           mobileCLIP = MobileCLIPInference(this)
           
           lifecycleScope.launch {{
               if (mobileCLIP.initialize()) {{
                   val labels = listOf("dog", "cat", "bird")
                   val results = mobileCLIP.zeroShotClassify(bitmap, labels)
                   
                   results?.take(3)?.forEach {{ (label, confidence) ->
                       Log.d("Classification", "$label: $confidence")
                   }}
               }}
           }}
       }}
       
       override fun onDestroy() {{
           super.onDestroy()
           mobileCLIP.close()
       }}
   }}
   ```

### Requirements
- Android API 24+
- TensorFlow Lite 2.13.0+
- Kotlin coroutines support

### Performance
- Inference time: ~80-450ms (device dependent)
- Memory usage: ~200-500MB
- NNAPI and GPU acceleration supported

### Optimization Tips
- Enable NNAPI acceleration for faster inference
- Use GPU delegate on supported devices
- Consider model quantization for smaller size

### Support
For issues and questions, refer to the main documentation.
"""
    
    with open(android_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    return str(android_dir)

def main():
    parser = argparse.ArgumentParser(description="Create MobileCLIP mobile deployment packages")
    parser.add_argument("--model", default="mobileclip_s0", help="Model name")
    parser.add_argument("--platforms", nargs="+", choices=["ios", "android"], 
                       default=["ios", "android"], help="Target platforms")
    parser.add_argument("--models-dir", default="./models/converted", help="Converted models directory")
    parser.add_argument("--output", default="./deployment/packages", help="Output directory")
    parser.add_argument("--zip", action="store_true", help="Create zip archives")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models_dir = Path(args.models_dir)
    
    print(f"📦 Creating deployment packages for {args.model}...")
    print(f"📁 Output directory: {output_dir}")
    print()
    
    packages = {}
    
    for platform in args.platforms:
        print(f"📱 Creating {platform.upper()} package...")
        
        try:
            if platform == "ios":
                # Look for CoreML models
                model_path = models_dir / f"{args.model}_image.mlpackage"
                if not model_path.exists():
                    model_path = models_dir / f"{args.model}.mlpackage"
                
                if model_path.exists():
                    package_path = create_ios_package(args.model, str(model_path), str(output_dir))
                    packages[platform] = package_path
                    print(f"✅ iOS package created: {package_path}")
                else:
                    print(f"❌ iOS model not found: {model_path}")
                    
            elif platform == "android":
                # Look for TensorFlow Lite models
                model_path = models_dir / f"{args.model}.tflite"
                
                if model_path.exists():
                    package_path = create_android_package(args.model, str(model_path), str(output_dir))
                    packages[platform] = package_path
                    print(f"✅ Android package created: {package_path}")
                else:
                    print(f"❌ Android model not found: {model_path}")
        
        except Exception as e:
            print(f"❌ Failed to create {platform} package: {e}")
    
    # Create zip archives if requested
    if args.zip and packages:
        print(f"\n📦 Creating zip archives...")
        
        for platform, package_path in packages.items():
            try:
                zip_path = Path(package_path).with_suffix('.zip')
                
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    package_dir = Path(package_path)
                    for file_path in package_dir.rglob('*'):
                        if file_path.is_file():
                            arcname = file_path.relative_to(package_dir.parent)
                            zipf.write(file_path, arcname)
                
                print(f"✅ {platform.upper()} archive: {zip_path}")
                
            except Exception as e:
                print(f"❌ Failed to create {platform} archive: {e}")
    
    # Print summary
    print("\n" + "="*50)
    print("DEPLOYMENT PACKAGES SUMMARY")
    print("="*50)
    
    for platform, package_path in packages.items():
        print(f"✅ {platform.upper()}: {package_path}")
        
        if args.zip:
            zip_path = Path(package_path).with_suffix('.zip')
            if zip_path.exists():
                size_mb = zip_path.stat().st_size / (1024 * 1024)
                print(f"   📦 Archive: {zip_path} ({size_mb:.1f} MB)")
    
    print(f"\n📊 Created {len(packages)} deployment packages")
    print("="*50)
    
    if not packages:
        print("❌ No packages were created. Check that converted models exist.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
