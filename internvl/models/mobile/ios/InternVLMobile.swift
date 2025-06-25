import CoreML
import Vision
import UIKit

@available(iOS 15.0, *)
public class InternVLMobile {
    private var model: MLModel?
    
    public init() {
        loadModel()
    }
    
    private func loadModel() {
        guard let modelURL = Bundle.main.url(forResource: "internvl_mobile", withExtension: "mlpackage") else {
            print("Failed to find model file (looking for internvl_mobile.mlpackage)")
            return
        }
        
        do {
            model = try MLModel(contentsOf: modelURL)
            print("Model loaded successfully")
        } catch {
            print("Failed to load model: \(error)")
        }
    }
    
    public func predict(pixelValues: MLMultiArray, inputIds: MLMultiArray, attentionMask: MLMultiArray) -> MLMultiArray? {
        guard let model = model else {
            print("Model not loaded")
            return nil
        }
        
        do {
            let input = InternVLMobileInput(
                pixel_values: pixelValues, 
                input_ids: inputIds, 
                attention_mask: attentionMask
            )
            let output = try model.prediction(from: input)
            
            if let result = output.featureValue(for: "output")?.multiArrayValue {
                return result
            }
        } catch {
            print("Prediction failed: \(error)")
        }
        
        return nil
    }
}

// Helper class for input
@available(iOS 15.0, *)
public class InternVLMobileInput: MLFeatureProvider {
    public var pixel_values: MLMultiArray
    public var input_ids: MLMultiArray
    public var attention_mask: MLMultiArray
    
    public init(pixel_values: MLMultiArray, input_ids: MLMultiArray, attention_mask: MLMultiArray) {
        self.pixel_values = pixel_values
        self.input_ids = input_ids
        self.attention_mask = attention_mask
    }
    
    public var featureNames: Set<String> {
        return ["pixel_values", "input_ids", "attention_mask"]
    }
    
    public func featureValue(for featureName: String) -> MLFeatureValue? {
        switch featureName {
        case "pixel_values":
            return MLFeatureValue(multiArray: pixel_values)
        case "input_ids":
            return MLFeatureValue(multiArray: input_ids)
        case "attention_mask":
            return MLFeatureValue(multiArray: attention_mask)
        default:
            return nil
        }
    }
}
