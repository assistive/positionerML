#!/bin/bash

# Unified VLM Mobile Converter - Easy execution script

echo "🚀 Unified VLM Mobile Converter"
echo "================================"

# Check if the main script exists
if [ ! -f "unified_mobile_converter.py" ]; then
    echo "❌ unified_mobile_converter.py not found!"
    echo "Please place the script in the current directory."
    exit 1
fi

# Set default options
PLATFORMS="ios android"
QUANTIZATION_BITS=8
ENABLE_PRUNING="--enable-pruning"
OUTPUT_DIR="mobile_models"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --ios-only)
            PLATFORMS="ios"
            shift
            ;;
        --android-only)
            PLATFORMS="android"
            shift
            ;;
        --4bit)
            QUANTIZATION_BITS=4
            shift
            ;;
        --16bit)
            QUANTIZATION_BITS=16
            shift
            ;;
        --no-pruning)
            ENABLE_PRUNING="--disable-pruning"
            shift
            ;;
        --discover)
            python unified_mobile_converter.py --discover-only
            exit 0
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --ios-only      Convert only for iOS"
            echo "  --android-only  Convert only for Android"
            echo "  --4bit         Use 4-bit quantization"
            echo "  --16bit        Use 16-bit quantization"
            echo "  --no-pruning   Disable model pruning"
            echo "  --discover     Only discover available models"
            echo "  --help         Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for available options"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the conversion
echo "Starting conversion with options:"
echo "  Platforms: $PLATFORMS"
echo "  Quantization: ${QUANTIZATION_BITS}-bit"
echo "  Pruning: $(echo $ENABLE_PRUNING | grep -q disable && echo 'disabled' || echo 'enabled')"
echo "  Output: $OUTPUT_DIR"
echo ""

python unified_mobile_converter.py \
    --all \
    --platforms $PLATFORMS \
    --quantization-bits $QUANTIZATION_BITS \
    $ENABLE_PRUNING \
    --output-dir "$OUTPUT_DIR" \
    --verbose

# Check results
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Conversion completed successfully!"
    echo "📁 Results saved in: $OUTPUT_DIR"
    echo "📊 Check conversion_report.json for detailed metrics"
    
    # Show quick summary
    echo ""
    echo "📱 Mobile models generated:"
    find "$OUTPUT_DIR" -name "*.onnx" -o -name "*.mlpackage" -o -name "*.tflite" | sort
else
    echo ""
    echo "❌ Conversion failed!"
    echo "📋 Check the logs for error details"
fi
