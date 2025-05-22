# SpatialLM iOS Integration Guide

This guide provides comprehensive instructions for integrating the SpatialLM model into an iOS application.

## Prerequisites

- Xcode 13.0 or higher
- iOS 15.0 or higher (minimum deployment target)
- Swift 5.5 or higher
- CocoaPods or Swift Package Manager for dependency management

## Project Setup

### 1. Add Dependencies

#### Using CocoaPods

Add the following to your Podfile:

```ruby
platform :ios, '15.0'

target 'YourAppName' do
  use_frameworks!
  
  # For Core ML integration
  pod 'CoreMLHelpers', '~> 0.6.0'
  
  # Natural Language Processing utilities
  pod 'NaturalLanguage'
  
  # For asynchronous operations
  pod 'PromiseKit', '~> 6.8'
end
```

Run `pod install` to install the dependencies.

#### Using Swift Package Manager

1. In Xcode, go to File > Swift Packages > Add Package Dependency
2. Add the CoreMLHelpers package: `https://github.com/hollance/CoreMLHelpers`
3. Add any other required packages

### 2. Add Core ML Model

1. Create a new group in your Xcode project for ML models
2. Drag and drop the `spatialLM_model.mlmodel` file into this group
3. Make sure "Copy items if needed" is checked
4. Add the model to your target

### 3. Add Tokenizer Files

1. Create a new group for tokenizer resources
2. Add the vocabulary files from the tokenizer directory
3. Make sure they are included in your app's bundle

## Implementation

### 1. Create SpatialLM Tokenizer Class

Create a Swift class to handle tokenization:

```swift
import Foundation
import NaturalLanguage

class SpatialLMTokenizer {
    
    private let vocabulary: [String: Int]
    private let idToToken: [Int: String]
    private let padToken: String
    private let eosToken: String
    private let unkToken: String
    
    init() throws {
        // Load vocabulary file
        guard let vocabURL = Bundle.main.url(forResource: "vocab", withExtension: "txt") else {
            throw TokenizerError.missingVocabFile
        }
        
        let vocabString = try String(contentsOf: vocabURL, encoding: .utf8)
        let tokens = vocabString.components(separatedBy: .newlines).filter { !$0.isEmpty }
        
        // Create token to ID mapping
        vocabulary = Dictionary(uniqueKeysWithValues: tokens.enumerated().map { ($1, $0) })
        
        // Create ID to token mapping
        idToToken = Dictionary(uniqueKeysWithValues: vocabulary.map { ($1, $0) })
        
        // Load model metadata to get special tokens
        guard let metadataURL = Bundle.main.url(forResource: "model_metadata", withExtension: "json") else {
            throw TokenizerError.missingMetadataFile
        }
        
        let metadataData = try Data(contentsOf: metadataURL)
        let metadata = try JSONSerialization.jsonObject(with: metadataData) as! [String: Any]
        let specialTokens = metadata["special_tokens"] as! [String: Any]
        
        padToken = specialTokens["pad_token"] as? String ?? "<pad>"
        eosToken = specialTokens["eos_token"] as? String ?? "</s>"
        unkToken = specialTokens["unk_token"] as? String ?? "<unk>"
    }
    
    func tokenize(text: String, maxLength: Int = 64) -> (inputIds: [Int], attentionMask: [Int]) {
        // Simple whitespace tokenization for demonstration
        // In a real implementation, you'd want to match the original tokenizer's behavior
        let tokens = text.split(separator: " ")
            .flatMap { $0.split(separator: "\n") }
            .filter { !$0.isEmpty }
            .map { String($0).lowercased() }
        
        // Convert tokens to IDs
        var inputIds = [Int]()
        for token in tokens {
            if let id = vocabulary[token] {
                inputIds.append(id)
            } else {
                // Use unknown token ID
                inputIds.append(vocabulary[unkToken] ?? 0)
            }
            
            if inputIds.count >= maxLength - 1 {
                break
            }
        }
        
        // Add EOS token
        if let eosId = vocabulary[eosToken] {
            inputIds.append(eosId)
        }
        
        // Create attention mask (1 for real tokens, 0 for padding)
        let attentionMask = Array(repeating: 1, count: maxLength)
        
        // Pad input IDs if necessary
        let padId = vocabulary[padToken] ?? 0
        let paddedInputIds = Array(inputIds.prefix(maxLength)) + Array(repeating: padId, count: max(0, maxLength - inputIds.count))
        
        return (paddedInputIds, attentionMask)
    }
    
    func decode(ids: [Int]) -> String {
        let tokens = ids.compactMap { idToToken[$0] }
            .filter { $0 != padToken && $0 != eosToken }
        
        return tokens.joined(separator: " ")
    }
    
    enum TokenizerError: Error {
        case missingVocabFile
        case missingMetadataFile
    }
}
```

### 2. Create SpatialLM Core ML Wrapper

Create a Swift class to handle model inference:

```swift
import Foundation
import CoreML
import NaturalLanguage

class SpatialLM {
    
    private let model: spatialLM_model
    private let tokenizer: SpatialLMTokenizer
    private let sequenceLength: Int
    private let spatialDim: Int
    
    init() throws {
        // Load Core ML model
        let config = MLModelConfiguration()
        config.computeUnits = .all
        
        model = try spatialLM_model(configuration: config)
        tokenizer = try SpatialLMTokenizer()
        
        // Load model metadata to get parameters
        guard let metadataURL = Bundle.main.url(forResource: "model_metadata", withExtension: "json") else {
            throw SpatialLMError.missingMetadataFile
        }
        
        let metadataData = try Data(contentsOf: metadataURL)
        let metadata = try JSONSerialization.jsonObject(with: metadataData) as! [String: Any]
        
        sequenceLength = metadata["sequence_length"] as? Int ?? 64
        spatialDim = metadata["spatial_dim"] as? Int ?? 3
    }
    
    func predictSpatial(text: String, coordinates: [Float]? = nil) throws -> [Float] {
        // Tokenize the input text
        let (inputIds, attentionMask) = tokenizer.tokenize(text: text, maxLength: sequenceLength)
        
        // Convert to MLMultiArray
        let inputIdsArray = try MLMultiArray(shape: [1, NSNumber(value: inputIds.count)], dataType: .int32)
        let attentionMaskArray = try MLMultiArray(shape: [1, NSNumber(value: attentionMask.count)], dataType: .int32)
        
        // Fill arrays with data
        for (index, id) in inputIds.enumerated() {
            inputIdsArray[index] = NSNumber(value: id)
        }
        
        for (index, mask) in attentionMask.enumerated() {
            attentionMaskArray[index] = NSNumber(value: mask)
        }
        
        // Create spatial coordinates array
        let spatialCoords = coordinates ?? Array(repeating: Float(0), count: spatialDim)
        let spatialCoordsArray = try MLMultiArray(shape: [1, NSNumber(value: spatialDim)], dataType: .float32)
        
        for (index, coord) in spatialCoords.enumerated() {
            spatialCoordsArray[index] = NSNumber(value: coord)
        }
        
        // Perform prediction
        let prediction = try model.prediction(
            input_ids: inputIdsArray,
            attention_mask: attentionMaskArray,
            spatial_coordinates: spatialCoordsArray
        )
        
        // Extract spatial predictions
        let spatialPredictions = (0..<spatialDim).map { Float(truncating: prediction.spatial_predictions[$0]) }
        
        return spatialPredictions
    }
    
    func generateText(promptText: String, maxNewTokens: Int = 20) throws -> String {
        // Tokenize the prompt
        var (inputIds, _) = tokenizer.tokenize(text: promptText, maxLength: sequenceLength)
        
        // Generate new tokens one by one
        for _ in 0..<maxNewTokens {
            // Ensure we don't exceed sequence length
            if inputIds.count >= sequenceLength {
                break
            }
            
            // Prepare input arrays
            let inputIdsArray = try MLMultiArray(shape: [1, NSNumber(value: inputIds.count)], dataType: .int32)
            let attentionMaskArray = try MLMultiArray(shape: [1, NSNumber(value: inputIds.count)], dataType: .int32)
            let spatialCoordsArray = try MLMultiArray(shape: [1, NSNumber(value: spatialDim)], dataType: .float32)
            
            // Fill arrays with data
            for (index, id) in inputIds.enumerated() {
                inputIdsArray[index] = NSNumber(value: id)
            }
            
            for index in 0..<inputIds.count {
                attentionMaskArray[index] = 1
            }
            
            // Perform prediction
            let prediction = try model.prediction(
                input_ids: inputIdsArray,
                attention_mask: attentionMaskArray,
                spatial_coordinates: spatialCoordsArray
            )
            
            // Get the logits for the last token
            let logitsCount = prediction.logits.count / inputIds.count
            var lastTokenLogits = [Float]()
            
            for i in 0..<logitsCount {
                let index = (inputIds.count - 1) * logitsCount + i
                lastTokenLogits.append(Float(truncating: prediction.logits[index]))
            }
            
            // Get the next token (simple greedy decoding)
            let nextTokenId = lastTokenLogits.enumerated().max(by: { $0.1 < $1.1 })?.0 ?? 0
            
            // Add the new token to the sequence
            inputIds.append(nextTokenId)
            
            // Stop if we generate EOS token
            if let eosId = tokenizer.vocabulary[tokenizer.eosToken], nextTokenId == eosId {
                break
            }
        }
        
        // Decode the generated sequence
        return tokenizer.decode(ids: inputIds)
    }
    
    enum SpatialLMError: Error {
        case missingMetadataFile
    }
}
```

### 3. Create a Repository Class

Create a repository class to handle SpatialLM operations:

```swift
import Foundation
import Combine

class SpatialLMRepository {
    
    private let spatialLM: SpatialLM
    
    init() throws {
        spatialLM = try SpatialLM()
    }
    
    func predictSpatialCoordinates(text: String) -> AnyPublisher<[Float], Error> {
        return Future<[Float], Error> { promise in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let coordinates = try self.spatialLM.predictSpatial(text: text)
                    promise(.success(coordinates))
                } catch {
                    promise(.failure(error))
                }
            }
        }.eraseToAnyPublisher()
    }
    
    func generateTextFromPrompt(prompt: String, maxNewTokens: Int = 20) -> AnyPublisher<String, Error> {
        return Future<String, Error> { promise in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let text = try self.spatialLM.generateText(promptText: prompt, maxNewTokens: maxNewTokens)
                    promise(.success(text))
                } catch {
                    promise(.failure(error))
                }
            }
        }.eraseToAnyPublisher()
    }
    
    func predictWithCoordinates(text: String, coordinates: [Float]) -> AnyPublisher<String, Error> {
        return Future<String, Error> { promise in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    // First, predict spatial information
                    let predictedCoordinates = try self.spatialLM.predictSpatial(text: text)
                    
                    // Then, generate text with the predicted coordinates
                    let enhancedPrompt = "\(text) [Coordinates: \(coordinates.map { String($0) }.joined(separator: ", "))]"
                    let generatedText = try self.spatialLM.generateText(promptText: enhancedPrompt)
                    promise(.success(generatedText))
                } catch {
                    promise(.failure(error))
                }
            }
        }.eraseToAnyPublisher()
    }
}
```

### 4. Create a ViewModel

Create a SwiftUI ViewModel to handle UI interactions:

```swift
import Foundation
import Combine
import SwiftUI

class SpatialLMViewModel: ObservableObject {
    
    private var repository: SpatialLMRepository
    private var cancellables = Set<AnyCancellable>()
    
    @Published var spatialPredictions: [Float] = []
    @Published var generatedText: String = ""
    @Published var isLoading: Bool = false
    @Published var errorMessage: String?
    
    init() {
        do {
            repository = try SpatialLMRepository()
        } catch {
            fatalError("Failed to initialize SpatialLM: \(error)")
        }
    }
    
    func predictSpatialCoordinates(text: String) {
        isLoading = true
        errorMessage = nil
        
        repository.predictSpatialCoordinates(text: text)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] completion in
                self?.isLoading = false
                
                if case .failure(let error) = completion {
                    self?.errorMessage = "Error: \(error.localizedDescription)"
                }
            } receiveValue: { [weak self] coordinates in
                self?.spatialPredictions = coordinates
            }
            .store(in: &cancellables)
    }
    
    func generateText(prompt: String) {
        isLoading = true
        errorMessage = nil
        
        repository.generateTextFromPrompt(prompt: prompt)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] completion in
                self?.isLoading = false
                
                if case .failure(let error) = completion {
                    self?.errorMessage = "Error: \(error.localizedDescription)"
                }
            } receiveValue: { [weak self] text in
                self?.generatedText = text
            }
            .store(in: &cancellables)
    }
}
```

### 5. Create SwiftUI View

Create a SwiftUI view for your interface:

```swift
import SwiftUI

struct SpatialLMView: View {
    
    @StateObject private var viewModel = SpatialLMViewModel()
    @State private var promptText = ""
    
    var body: some View {
        VStack(spacing: 16) {
            TextField("Enter text prompt", text: $promptText, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(3)
                .padding(.horizontal)
            
            HStack {
                Button("Predict Spatial") {
                    if !promptText.isEmpty {
                        viewModel.predictSpatialCoordinates(text: promptText)
                    }
                }
                .buttonStyle(.borderedProminent)
                
                Button("Generate Text") {
                    if !promptText.isEmpty {
                        viewModel.generateText(prompt: promptText)
                    }
                }
                .buttonStyle(.borderedProminent)
            }
            .padding(.horizontal)
            
            if !viewModel.spatialPredictions.isEmpty {
                VStack(alignment: .leading) {
                    Text("Predicted Coordinates:")
                        .font(.headline)
                    
                    Text(formatCoordinates(viewModel.spatialPredictions))
                        .padding()
                        .background(Color.gray.opacity(0.1))
                        .cornerRadius(8)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal)
            }
            
            if !viewModel.generatedText.isEmpty {
                VStack(alignment: .leading) {
                    Text("Generated Text:")
                        .font(.headline)
                    
                    ScrollView {
                        Text(viewModel.generatedText)
                            .padding()
                            .background(Color.gray.opacity(0.1))
                            .cornerRadius(8)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal)
            }
            
            if let errorMessage = viewModel.errorMessage {
                Text(errorMessage)
                    .foregroundColor(.red)
                    .padding()
            }
            
            Spacer()
        }
        .padding(.vertical)
        .overlay {
            if viewModel.isLoading {
                ProgressView()
                    .scaleEffect(1.5)
                    .background(Color.white.opacity(0.7))
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .navigationTitle("SpatialLM Demo")
    }
    
    private func formatCoordinates(_ coordinates: [Float]) -> String {
        switch coordinates.count {
        case 3:
            return "X: \(coordinates[0]), Y: \(coordinates[1]), Z: \(coordinates[2])"
        case 2:
            return "X: \(coordinates[0]), Y: \(coordinates[1])"
        case 1:
            return "Value: \(coordinates[0])"
        default:
            return coordinates.map { String(format: "%.4f", $0) }.joined(separator: ", ")
        }
    }
}

struct SpatialLMView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationView {
            SpatialLMView()
        }
    }
}
```

## Best Practices

### 1. Model Loading Optimization

- Load the Core ML model asynchronously to avoid blocking the main thread
- Use background queues for model loading and inference
- Initialize the model once and reuse it throughout the app

### 2. Memory Management

- Core ML models can be memory-intensive, so be mindful of memory usage
- Use the `purgeDataSource` method on the model if memory pressure occurs
- Consider releasing the model when it's not in use, especially for large models

### 3. Performance Optimization

- Use the `.all` compute units option to enable Neural Engine acceleration
- For devices without Neural Engine, fall back to CPU+GPU
- Consider using a smaller, quantized model for better performance
- Use batched predictions when processing multiple inputs

### 4. Battery Efficiency

- Run model inference on a background thread
- Avoid unnecessary model evaluations
- Consider implementing throttling for continuous predictions

### 5. Model Versioning

- Implement a versioning scheme for your models
- Use on-device model updates via Core ML Model Deployment

## Troubleshooting

### Common Issues

1. **Model Loading Failures**
   
   *Solution*: Verify the model file is correctly added to the app bundle and check deployment target compatibility.

2. **Out of Memory Errors**
   
   *Solution*: Reduce batch size, use a smaller model, or implement memory management strategies.

3. **Slow Inference**
   
   *Solution*: Enable Neural Engine, optimize inputs/outputs, or use a quantized model.

4. **Incorrect Tokenization**
   
   *Solution*: Ensure your tokenization logic matches the one used during training.

5. **Neural Engine Compatibility**
   
   *Solution*: Some models may not be compatible with the Neural Engine. Use the `.cpuAndGPU` compute units as a fallback.

## Advanced Topics

### On-Device Model Updates

You can implement on-device model updates using Core ML Model Deployment:

```swift
import CoreML

func updateModel() {
    let modelURL = URL(string: "https://your-server.com/models/spatialLM_model.mlmodel")!
    
    let task = URLSession.shared.downloadTask(with: modelURL) { tempURL, response, error in
        guard let tempURL = tempURL, error == nil else {
            print("Download failed: \(error?.localizedDescription ?? "Unknown error")")
            return
        }
        
        do {
            // Get the app's document directory
            let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let destinationURL = documentsDirectory.appendingPathComponent("spatialLM_model.mlmodel")
            
            // Remove any existing model file
            if FileManager.default.fileExists(atPath: destinationURL.path) {
                try FileManager.default.removeItem(at: destinationURL)
            }
            
            // Move the downloaded model to the documents directory
            try FileManager.default.moveItem(at: tempURL, to: destinationURL)
            
            // Compile the model
            let compiledURL = try MLModel.compileModel(at: destinationURL)
            
            // Load the updated model
            let config = MLModelConfiguration()
            config.computeUnits = .all
            
            let updatedModel = try MLModel(contentsOf: compiledURL, configuration: config)
            
            // Update your model reference
            DispatchQueue.main.async {
                // Update your model instance
                print("Model successfully updated!")
            }
        } catch {
            print("Error updating model: \(error.localizedDescription)")
        }
    }
    
    task.resume()
}
```

### Custom Compute Units Configuration

For more fine-grained control over model execution:

```swift
let config = MLModelConfiguration()

// For best performance
config.computeUnits = .all

// For battery efficiency
config.computeUnits = .cpuOnly

// For devices without Neural Engine
config.computeUnits = .cpuAndGPU

// Load model with configuration
let model = try spatialLM_model(configuration: config)
```

## Resources

- [Core ML Documentation](https://developer.apple.com/documentation/coreml)
- [Core ML Tools](https://coremltools.readme.io/docs)
- [Core ML Performance Best Practices](https://developer.apple.com/documentation/coreml/core_ml_performance)
- [GitHub Sample Project](https://github.com/spatialLM/spatialLM-ios-demo)
