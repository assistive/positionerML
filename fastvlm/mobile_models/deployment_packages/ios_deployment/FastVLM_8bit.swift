import CoreML
import Vision
import UIKit

@available(iOS 15.0, *)
public class FastVLM {
    private let model: MLModel
    private let imageSize: Int = 224
    private let maxSequenceLength: Int = 512
    
    public init() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all // Use Neural Engine when available
        config.allowLowPrecisionAccumulationOnGPU = true
        
        guard let modelURL = Bundle.main.url(forResource: "FastVLM_8bit", withExtension: "mlpackage") else {
            throw FastVLMError.modelNotFound
        }
        
        self.model = try MLModel(contentsOf: modelURL, configuration: config)
    }
    
    public func predict(image: UIImage, prompt: String) async throws -> String {
        // Preprocess image
        guard let pixelBuffer = image.pixelBuffer(width: imageSize, height: imageSize) else {
            throw FastVLMError.imageProcessingFailed
        }
        
        // Tokenize text
        let tokens = tokenize(text: prompt, maxLength: maxSequenceLength)
        let inputIds = try MLMultiArray(shape: [1, maxSequenceLength], dataType: .int32)
        let attentionMask = try MLMultiArray(shape: [1, maxSequenceLength], dataType: .int32)
        
        // Fill arrays
        for i in 0..<tokens.count {
            inputIds[i] = NSNumber(value: tokens[i])
            attentionMask[i] = NSNumber(value: 1)
        }
        
        // Create input features
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(pixelBuffer: pixelBuffer),
            "input_ids": MLFeatureValue(multiArray: inputIds),
            "attention_mask": MLFeatureValue(multiArray: attentionMask)
        ])
        
        // Run prediction
        let output = try await model.prediction(from: input)
        
        // Decode output
        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw FastVLMError.predictionFailed
        }
        
        return decodeOutput(logits)
    }
    
    private func tokenize(text: String, maxLength: Int) -> [Int32] {
        // Simple tokenization (replace with proper tokenizer)
        let words = text.lowercased().components(separatedBy: .whitespacesAndNewlines)
        var tokens: [Int32] = [101] // CLS token
        
        for word in words.prefix(maxLength - 2) {
            tokens.append(Int32(word.hashValue % 30000 + 1000))
        }
        
        tokens.append(102) // SEP token
        
        // Pad to maxLength
        while tokens.count < maxLength {
            tokens.append(0) // PAD token
        }
        
        return Array(tokens.prefix(maxLength))
    }
    
    private func decodeOutput(_ logits: MLMultiArray) -> String {
        // Simple decoding (replace with proper decoder)
        return "Generated response based on image and prompt"
    }
}

public enum FastVLMError: Error {
    case modelNotFound
    case imageProcessingFailed
    case predictionFailed
}

extension UIImage {
    func pixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                        width, height,
                                        kCVPixelFormatType_32ARGB,
                                        attrs, &pixelBuffer)
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(buffer)
        
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: pixelData,
                               width: width, height: height,
                               bitsPerComponent: 8,
                               bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                               space: rgbColorSpace,
                               bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
        
        context?.translateBy(x: 0, y: CGFloat(height))
        context?.scaleBy(x: 1.0, y: -1.0)
        
        UIGraphicsPushContext(context!)
        self.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
        UIGraphicsPopContext()
        CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        
        return buffer
    }
}
