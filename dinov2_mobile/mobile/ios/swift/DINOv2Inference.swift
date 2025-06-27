import CoreML
import Vision
import UIKit
import Metal
import MetalPerformanceShaders

@available(iOS 15.0, *)
class DINOv2Inference {
    
    private var model: MLModel?
    private var visionModel: VNCoreMLModel?
    
    static let shared = DINOv2Inference()
    
    private init() {
        loadModel()
    }
    
    private func loadModel() {
        guard let modelURL = Bundle.main.url(forResource: "dinov2_mobile", withExtension: "mlpackage") else {
            print("❌ Could not find dinov2_mobile.mlpackage in bundle")
            return
        }
        
        do {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .all // Use Neural Engine when available
            
            model = try MLModel(contentsOf: modelURL, configuration: configuration)
            visionModel = try VNCoreMLModel(for: model!)
            
            print("✅ DINOv2 model loaded successfully")
        } catch {
            print("❌ Failed to load model: \(error)")
        }
    }
    
    func extractFeatures(from image: UIImage, completion: @escaping (Result<[Float], Error>) -> Void) {
        guard let visionModel = visionModel else {
            completion(.failure(DINOv2Error.modelNotLoaded))
            return
        }
        
        guard let cgImage = image.cgImage else {
            completion(.failure(DINOv2Error.invalidImage))
            return
        }
        
        let request = VNCoreMLRequest(model: visionModel) { request, error in
            if let error = error {
                completion(.failure(error))
                return
            }
            
            guard let results = request.results as? [VNCoreMLFeatureValueObservation],
                  let firstResult = results.first,
                  let features = firstResult.featureValue.multiArrayValue else {
                completion(.failure(DINOv2Error.featureExtractionFailed))
                return
            }
            
            // Convert MLMultiArray to Float array
            let featureArray = self.convertToFloatArray(features)
            completion(.success(featureArray))
        }
        
        // Configure request
        request.imageCropAndScaleOption = .centerCrop
        
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let startTime = CFAbsoluteTimeGetCurrent()
                try handler.perform([request])
                let inferenceTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
                print("🔧 Inference time: \(String(format: "%.1f", inferenceTime))ms")
            } catch {
                completion(.failure(error))
            }
        }
    }
    
    private func convertToFloatArray(_ multiArray: MLMultiArray) -> [Float] {
        let count = multiArray.count
        var floatArray = [Float](repeating: 0, count: count)
        
        for i in 0..<count {
            floatArray[i] = Float(truncating: multiArray[i])
        }
        
        return floatArray
    }
    
    func compareFeatures(_ features1: [Float], _ features2: [Float]) -> Float {
        guard features1.count == features2.count else { return 0.0 }
        
        // Calculate cosine similarity
        let dotProduct = zip(features1, features2).reduce(0) { $0 + $1.0 * $1.1 }
        let norm1 = sqrt(features1.reduce(0) { $0 + $1 * $1 })
        let norm2 = sqrt(features2.reduce(0) { $0 + $1 * $1 })
        
        return dotProduct / (norm1 * norm2)
    }
}

enum DINOv2Error: Error {
    case modelNotLoaded
    case invalidImage
    case featureExtractionFailed
    
    var localizedDescription: String {
        switch self {
        case .modelNotLoaded:
            return "DINOv2 model is not loaded"
        case .invalidImage:
            return "Invalid image provided"
        case .featureExtractionFailed:
            return "Feature extraction failed"
        }
    }
}

// MARK: - Utility Extensions
extension UIImage {
    func resized(to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, 0.0)
        draw(in: CGRect(origin: .zero, size: size))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return resizedImage
    }
}
