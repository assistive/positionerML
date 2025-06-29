//
//  MobileCLIPInference.swift
//  MobileCLIP iOS Integration
//
//  Provides easy-to-use interface for MobileCLIP inference on iOS
//

import Foundation
import CoreML
import Vision
import UIKit
import Accelerate

@available(iOS 15.0, *)
public class MobileCLIPInference: ObservableObject {
    
    // MARK: - Properties
    private var imageModel: MLModel?
    private var textModel: MLModel?
    private let imageSize = CGSize(width: 224, height: 224)
    
    // MARK: - Singleton
    public static let shared = MobileCLIPInference()
    
    private init() {
        loadModels()
    }
    
    // MARK: - Model Loading
    private func loadModels() {
        do {
            // Load image encoder model
            if let imageModelURL = Bundle.main.url(forResource: "mobileclip_image", withExtension: "mlpackage") {
                imageModel = try MLModel(contentsOf: imageModelURL)
                print("✅ Image model loaded successfully")
            }
            
            // Load text encoder model
            if let textModelURL = Bundle.main.url(forResource: "mobileclip_text", withExtension: "mlpackage") {
                textModel = try MLModel(contentsOf: textModelURL)
                print("✅ Text model loaded successfully")
            }
        } catch {
            print("❌ Failed to load models: \(error)")
        }
    }
    
    // MARK: - Image Processing
    private func preprocessImage(_ image: UIImage) -> CVPixelBuffer? {
        // Resize image to 224x224
        guard let resizedImage = image.resized(to: imageSize) else {
            print("Failed to resize image")
            return nil
        }
        
        // Convert to CVPixelBuffer
        return resizedImage.toCVPixelBuffer()
    }
    
    // MARK: - Text Processing
    private func tokenizeText(_ text: String) -> [Int32] {
        // Simple tokenization - in practice, use proper tokenizer
        let maxLength = 77
        var tokens = Array(repeating: Int32(0), count: maxLength)
        
        // Convert text to token IDs (simplified)
        let words = text.lowercased().components(separatedBy: .whitespacesAndNewlines)
        for (index, word) in words.enumerated() {
            if index < maxLength - 2 {
                tokens[index + 1] = Int32(word.hash % 30000) // Simplified hashing
            }
        }
        
        tokens[0] = 49406 // Start token
        if words.count < maxLength - 2 {
            tokens[words.count + 1] = 49407 // End token
        }
        
        return tokens
    }
    
    // MARK: - Inference
    public func extractImageFeatures(from image: UIImage, completion: @escaping (Result<[Float], Error>) -> Void) {
        guard let imageModel = imageModel else {
            completion(.failure(MobileCLIPError.modelNotLoaded))
            return
        }
        
        guard let pixelBuffer = preprocessImage(image) else {
            completion(.failure(MobileCLIPError.imageProcessingFailed))
            return
        }
        
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let input = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: pixelBuffer)])
                let output = try imageModel.prediction(from: input)
                
                if let features = output.featureValue(for: "image_features")?.multiArrayValue {
                    let floatArray = self.multiArrayToFloatArray(features)
                    DispatchQueue.main.async {
                        completion(.success(floatArray))
                    }
                } else {
                    DispatchQueue.main.async {
                        completion(.failure(MobileCLIPError.inferenceError))
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
            }
        }
    }
    
    public func extractTextFeatures(from text: String, completion: @escaping (Result<[Float], Error>) -> Void) {
        guard let textModel = textModel else {
            completion(.failure(MobileCLIPError.modelNotLoaded))
            return
        }
        
        let tokens = tokenizeText(text)
        
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let tokensArray = try MLMultiArray(shape: [1, 77], dataType: .int32)
                for (index, token) in tokens.enumerated() {
                    tokensArray[index] = NSNumber(value: token)
                }
                
                let input = try MLDictionaryFeatureProvider(dictionary: ["text_tokens": MLFeatureValue(multiArray: tokensArray)])
                let output = try textModel.prediction(from: input)
                
                if let features = output.featureValue(for: "text_features")?.multiArrayValue {
                    let floatArray = self.multiArrayToFloatArray(features)
                    DispatchQueue.main.async {
                        completion(.success(floatArray))
                    }
                } else {
                    DispatchQueue.main.async {
                        completion(.failure(MobileCLIPError.inferenceError))
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
            }
        }
    }
    
    public func computeSimilarity(imageFeatures: [Float], textFeatures: [Float]) -> Float {
        guard imageFeatures.count == textFeatures.count else {
            print("Feature vector sizes don't match")
            return 0.0
        }
        
        // Compute cosine similarity
        var dotProduct: Float = 0.0
        var imageNorm: Float = 0.0
        var textNorm: Float = 0.0
        
        for i in 0..<imageFeatures.count {
            dotProduct += imageFeatures[i] * textFeatures[i]
            imageNorm += imageFeatures[i] * imageFeatures[i]
            textNorm += textFeatures[i] * textFeatures[i]
        }
        
        let similarity = dotProduct / (sqrt(imageNorm) * sqrt(textNorm))
        return similarity
    }
    
    public func zeroShotClassify(image: UIImage, labels: [String], completion: @escaping (Result<[(label: String, confidence: Float)], Error>) -> Void) {
        extractImageFeatures(from: image) { [weak self] imageResult in
            switch imageResult {
            case .success(let imageFeatures):
                let group = DispatchGroup()
                var textFeatures: [[Float]] = []
                var errors: [Error] = []
                
                for label in labels {
                    group.enter()
                    self?.extractTextFeatures(from: label) { textResult in
                        switch textResult {
                        case .success(let features):
                            textFeatures.append(features)
                        case .failure(let error):
                            errors.append(error)
                        }
                        group.leave()
                    }
                }
                
                group.notify(queue: .main) {
                    guard errors.isEmpty else {
                        completion(.failure(errors.first!))
                        return
                    }
                    
                    var results: [(label: String, confidence: Float)] = []
                    for (index, features) in textFeatures.enumerated() {
                        let similarity = self?.computeSimilarity(imageFeatures: imageFeatures, textFeatures: features) ?? 0.0
                        results.append((label: labels[index], confidence: similarity))
                    }
                    
                    // Sort by confidence
                    results.sort { $0.confidence > $1.confidence }
                    completion(.success(results))
                }
                
            case .failure(let error):
                completion(.failure(error))
            }
        }
    }
    
    // MARK: - Utility Methods
    private func multiArrayToFloatArray(_ multiArray: MLMultiArray) -> [Float] {
        let count = multiArray.count
        var floatArray = Array<Float>(repeating: 0, count: count)
        
        let dataPointer = multiArray.dataPointer.bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            floatArray[i] = dataPointer[i]
        }
        
        return floatArray
    }
}

// MARK: - Error Types
public enum MobileCLIPError: Error, LocalizedError {
    case modelNotLoaded
    case imageProcessingFailed
    case inferenceError
    
    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "MobileCLIP model not loaded"
        case .imageProcessingFailed:
            return "Failed to process image"
        case .inferenceError:
            return "Inference failed"
        }
    }
}

// MARK: - UIImage Extensions
extension UIImage {
    func resized(to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, 0.0)
        draw(in: CGRect(origin: .zero, size: size))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return resizedImage
    }
    
    func toCVPixelBuffer() -> CVPixelBuffer? {
        let width = Int(self.size.width)
        let height = Int(self.size.height)
        
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                       width,
                                       height,
                                       kCVPixelFormatType_32ARGB,
                                       attrs,
                                       &pixelBuffer)
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(buffer)
        
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: pixelData,
                              width: width,
                              height: height,
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
