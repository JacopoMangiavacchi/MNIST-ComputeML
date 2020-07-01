//
//  MNIST.swift
//  CoreML_Training
//
//  Created by Jacopo Mangiavacchi on 6/28/20.
//  Copyright © 2020 Jacopo Mangiavacchi. All rights reserved.
//

import Foundation
import MLCompute

public class MNIST : ObservableObject {
    let imageSize = 28*28
    let concurrentQueue = DispatchQueue(label: "MNIST.concurrent.queue", attributes: .concurrent)

    public enum BatchPreparationStatus {
        case notPrepared
        case preparing(count: Int)
        case ready
        
        var description: String {
            switch self {
            case .notPrepared:
                return "Not Prepared"
            case .preparing(let count):
                return "Preparing \(count)"
            case .ready:
                return "Ready"
            }
        }
    }
    
    @Published public var trainingBatchStatus = BatchPreparationStatus.notPrepared
    @Published public var predictionBatchStatus = BatchPreparationStatus.notPrepared
    @Published public var trainingBatchProviderXTensor: MLCTensor?
    @Published public var trainingBatchProviderYTensor: MLCTensor?
    @Published public var predictionBatchProviderXTensor: MLCTensor?
    @Published public var predictionBatchProviderYTensor: MLCTensor?
    @Published public var modelPrepared = false
    @Published public var modelCompiled = false
    @Published public var modelTrained = false
    @Published public var modelStatus = "Train model"
    @Published public var accuracy = "Accuracy: n/a"
    @Published public var epoch: Int = 5

    
//    var coreMLModelUrl: URL
//    var coreMLCompiledModelUrl: URL?
//    var model: MLModel?
//    var retrainedModel: MLModel?
//    var predictionLabels: [Int]
//    var trainingStartTime: Date!
    
//    public init() {
//        predictionLabels = [Int]()
//    }
    
    public func readDataSet(fileName: String, updateStatus: (Int) -> Void) -> (MLCTensor, MLCTensor) {
        var count = 0
        var X = [Float]()
        var Y = [Int64]()

        guard let filePath = Bundle.main.path(forResource: fileName, ofType: "csv") else {
            fatalError("CSV file not found")
        }
        do {
            let content = try String(contentsOfFile: filePath)
            for line in content.split(whereSeparator: \.isNewline) {
                let sample = line.split(separator: ",").compactMap({Int64(String($0))})
                Y.append(sample[0])
                X.append(contentsOf: sample[1...imageSize].map{Float($0) / Float(255.0)})

                count += 1
                updateStatus(count)
            }
        } catch {
            fatalError("Error reading CSV")
        }
        
        let xData = X.withUnsafeBufferPointer { pointer in
            MLCTensorData(immutableBytesNoCopy: pointer.baseAddress!,
                          length: pointer.count * MemoryLayout<Float>.size * imageSize)
        }

        let yData = Y.withUnsafeBufferPointer { pointer in
            MLCTensorData(immutableBytesNoCopy: pointer.baseAddress!,
                          length: pointer.count * MemoryLayout<Int>.size)
        }

        let xTensor = MLCTensor(descriptor: MLCTensorDescriptor(shape: [count, imageSize], dataType: .float32)!,
                                data: xData)

        let yTensor = MLCTensor(descriptor: MLCTensorDescriptor(shape: [count, 1], dataType: .int64)!,
                                data: yData)

        return (xTensor, yTensor)
    }
        
    public func asyncPrepareTrainBatchProvider() {
        self.trainingBatchStatus = .preparing(count: 0)
        concurrentQueue.async {
            let (X, Y) = self.readDataSet(fileName: "mnist_train") { count in
                DispatchQueue.main.async {
                    self.trainingBatchStatus = .preparing(count: count)
                }
            }
            
            DispatchQueue.main.async {
                self.trainingBatchStatus = .ready
                self.trainingBatchProviderXTensor = X
                self.trainingBatchProviderYTensor = Y
            }
        }
    }
    
    public func asyncPreparePredictionBatchProvider() {
        self.predictionBatchStatus = .preparing(count: 0)
        concurrentQueue.async {
            let (X, Y) = self.readDataSet(fileName: "mnist_test") { count in
                DispatchQueue.main.async {
                    self.predictionBatchStatus = .preparing(count: count)
                }
            }
            
            DispatchQueue.main.async {
                self.predictionBatchStatus = .ready
                self.predictionBatchProviderXTensor = X
                self.predictionBatchProviderYTensor = Y
            }
        }
    }
    
    public func prepareGraph() -> MLCGraph {
        
        let graph = MLCGraph()
//        let tensor1plus2 = graph.node(with: MLCArithmeticLayer(operation: .add), sources: [tensor1, tensor2])
//        graph.node(with: MLCArithmeticLayer(operation: .add), sources: [tensor1plus2!, tensor3])
        
        // DENSE LAYER
        // -----------
        //  INPUT SHAPE: (768, 1)
        //  LABEL SHAPE: (10, 1)
        //  OUTPUT SHAPE:
        let dense = graph.node(with: MLCFullyConnectedLayer(weights: MLCTensor(descriptor: MLCTensorDescriptor(shape: [6, 1], dataType: .float32)!),
                                                            biases: MLCTensor(descriptor: MLCTensorDescriptor(shape: [6, 1], dataType: .float32)!),
                                                            descriptor: MLCConvolutionDescriptor(kernelSizes: (height: 6, width: 1),
                                                                                                 inputFeatureChannelCount: 6,
                                                                                                 outputFeatureChannelCount: 1))!,
                               sources: [MLCTensor(descriptor: MLCTensorDescriptor(shape: [6, 1], dataType: .float32)!)],
                               lossLabels: [MLCTensor(descriptor: MLCTensorDescriptor(shape: [6, 1], dataType: .float32)!)])
        
        // SOFTMAX ACTIVATION
        // ------------------
        graph.node(with: MLCSoftmaxLayer(operation: MLCSoftmaxOperation(rawValue: 10)!),
                   source: dense!)
        
        
        
        return graph

        
//        let coremlModel = Model(version: 4,
//                                shortDescription: "MNIST-Trainable",
//                                author: "Jacopo Mangiavacchi",
//                                license: "MIT",
//                                userDefined: ["SwiftCoremltoolsVersion" : "0.0.12"]) {
//            Input(name: "image", shape: [1, 28, 28])
//            Output(name: "output", shape: [10], featureType: .float)
//            TrainingInput(name: "image", shape: [1, 28, 28])
//            TrainingInput(name: "output_true", shape: [1], featureType: .int)
//            NeuralNetwork(losses: [CategoricalCrossEntropy(name: "lossLayer",
//                                       input: "output",
//                                       target: "output_true")],
//                          optimizer: Adam(learningRateDefault: 0.0001,
//                                         learningRateMax: 0.3,
//                                         miniBatchSizeDefault: 128,
//                                         miniBatchSizeRange: [128],
//                                         beta1Default: 0.9,
//                                         beta1Max: 1.0,
//                                         beta2Default: 0.999,
//                                         beta2Max: 1.0,
//                                         epsDefault: 0.00000001,
//                                         epsMax: 0.00000001),
//                          epochDefault: UInt(self.epoch),
//                          epochSet: [UInt(self.epoch)],
//                          shuffle: true) {
//                Convolution(name: "conv1",
//                             input: ["image"],
//                             output: ["outConv1"],
//                             outputChannels: 32,
//                             kernelChannels: 1,
//                             nGroups: 1,
//                             kernelSize: [3, 3],
//                             stride: [1, 1],
//                             dilationFactor: [1, 1],
//                             paddingType: .valid(borderAmounts: [EdgeSizes(startEdgeSize: 0, endEdgeSize: 0),
//                                                                 EdgeSizes(startEdgeSize: 0, endEdgeSize: 0)]),
//                             outputShape: [],
//                             deconvolution: false,
//                             updatable: true)
//                ReLu(name: "relu1",
//                     input: ["outConv1"],
//                     output: ["outRelu1"])
//                Pooling(name: "pooling1",
//                             input: ["outRelu1"],
//                             output: ["outPooling1"],
//                             poolingType: .max,
//                             kernelSize: [2, 2],
//                             stride: [2, 2],
//                             paddingType: .valid(borderAmounts: [EdgeSizes(startEdgeSize: 0, endEdgeSize: 0),
//                                                                 EdgeSizes(startEdgeSize: 0, endEdgeSize: 0)]),
//                             avgPoolExcludePadding: true,
//                             globalPooling: false)
//                Convolution(name: "conv2",
//                             input: ["outPooling1"],
//                             output: ["outConv2"],
//                             outputChannels: 32,
//                             kernelChannels: 32,
//                             nGroups: 1,
//                             kernelSize: [2, 2],
//                             stride: [1, 1],
//                             dilationFactor: [1, 1],
//                             paddingType: .valid(borderAmounts: [EdgeSizes(startEdgeSize: 0, endEdgeSize: 0),
//                                                                 EdgeSizes(startEdgeSize: 0, endEdgeSize: 0)]),
//                             outputShape: [],
//                             deconvolution: false,
//                             updatable: true)
//                ReLu(name: "relu2",
//                     input: ["outConv2"],
//                     output: ["outRelu2"])
//                Pooling(name: "pooling2",
//                             input: ["outRelu2"],
//                             output: ["outPooling2"],
//                             poolingType: .max,
//                             kernelSize: [2, 2],
//                             stride: [2, 2],
//                             paddingType: .valid(borderAmounts: [EdgeSizes(startEdgeSize: 0, endEdgeSize: 0),
//                                                                 EdgeSizes(startEdgeSize: 0, endEdgeSize: 0)]),
//                             avgPoolExcludePadding: true,
//                             globalPooling: false)
//                Flatten(name: "flatten1",
//                             input: ["outPooling2"],
//                             output: ["outFlatten1"],
//                             mode: .last)
//                InnerProduct(name: "hidden1",
//                             input: ["outFlatten1"],
//                             output: ["outHidden1"],
//                             inputChannels: 1152,
//                             outputChannels: 500,
//                             updatable: true)
//                ReLu(name: "relu3",
//                     input: ["outHidden1"],
//                     output: ["outRelu3"])
//                InnerProduct(name: "hidden2",
//                             input: ["outRelu3"],
//                             output: ["outHidden2"],
//                             inputChannels: 500,
//                             outputChannels: 10,
//                             updatable: true)
//                Softmax(name: "softmax",
//                        input: ["outHidden2"],
//                        output: ["output"])
//            }
//        }
//
//        let coreMLData = coremlModel.coreMLData
//        print(coreMLModelUrl)
//        try! coreMLData!.write(to: coreMLModelUrl)
//        modelPrepared = true
    }
    
//    public func compileModel() {
//        coreMLCompiledModelUrl = try! MLModel.compileModel(at: coreMLModelUrl)
//        print("Compiled Model Path: \(coreMLCompiledModelUrl!)")
//        model = try! MLModel(contentsOf: coreMLCompiledModelUrl!)
//        modelCompiled = true
//    }
//
//    public func trainModel() {
//        self.modelTrained = false
//        self.modelStatus = "Training starting"
//
//        let configuration = MLModelConfiguration()
//        configuration.computeUnits = .all
//        //configuration.parameters = [.epochs : 100]
//        let progressHandler = { (context: MLUpdateContext) in
//            switch context.event {
//            case .trainingBegin:
//                print("Training started..")
//                DispatchQueue.main.async {
//                    self.modelStatus = "Training started.."
//                }
//
//            case .miniBatchEnd:
//                break
////                let batchIndex = context.metrics[.miniBatchIndex] as! Int
////                let batchLoss = context.metrics[.lossValue] as! Double
////                print("Mini batch \(batchIndex), loss: \(batchLoss)")
//            case .epochEnd:
//                let epochIndex = context.metrics[.epochIndex] as! Int
//                let trainLoss = context.metrics[.lossValue] as! Double
//                print("Epoch \(epochIndex + 1) end with loss \(trainLoss)")
//                DispatchQueue.main.async {
//                    self.modelStatus = "Epoch \(epochIndex) end with loss \(trainLoss)"
//                }
//
//            default:
//                print("Unknown event")
//            }
//
////        print(context.model.modelDescription.parameterDescriptionsByKey)
////        do {
////            let multiArray = try context.model.parameterValue(for: MLParameterKey.weights.scoped(to: "dense_1")) as! MLMultiArray
////            print(multiArray.shape)
////        } catch {
////            print(error)
////        }
//        }
//
//        let completionHandler = { (context: MLUpdateContext) in
//            print("Training completed with state \(context.task.state.rawValue)")
//            print("CoreML Error: \(context.task.error.debugDescription)")
//            DispatchQueue.main.async {
//                self.modelStatus = "Training completed with state \(context.task.state.rawValue)"
//            }
//
//            if context.task.state != .completed {
//                print("Failed")
//                DispatchQueue.main.async {
//                    self.modelStatus = "Training Failed"
//                }
//                return
//            }
//
//            let trainLoss = context.metrics[.lossValue] as! Double
//            print("Final loss: \(trainLoss)")
//            DispatchQueue.main.async {
//                self.modelStatus = "Training completed with loss: \(trainLoss) in \(Int(Date().timeIntervalSince(self.trainingStartTime))) secs"
//                self.modelTrained = true
//            }
//
//            self.retrainedModel = context.model
//
////            let updatedModel = context.model
////            let updatedModelURL = URL(fileURLWithPath: retrainedCoreMLFilePath)
////            try! updatedModel.write(to: updatedModelURL)
//            print("Model Trained!")
//        }
//
//        let handlers = MLUpdateProgressHandlers(
//                            forEvents: [.trainingBegin, .miniBatchEnd, .epochEnd],
//                            progressHandler: progressHandler,
//                            completionHandler: completionHandler)
//
//        self.trainingStartTime = Date()
//
//        let updateTask = try! MLUpdateTask(forModelAt: coreMLCompiledModelUrl!,
//                                           trainingData: trainingBatchProvider!,
//                                           configuration: configuration,
//                                           progressHandlers: handlers)
//
//        updateTask.resume()
//    }
//
//    public func testModel() {
//        let predictionProvider = try! self.retrainedModel?.predictions(fromBatch: predictionBatchProvider!)
//
//        print(predictionProvider!.count)
//        var correct = 0
//        for i in 0..<predictionProvider!.count {
//            let label = predictionLabels[i]
//            let predictionEncoded = predictionProvider!.features(at: i).featureValue(for: "output")!
//
//            if predictionEncoded.multiArrayValue![label].floatValue > 0.5 {
//                correct += 1
//            }
//        }
//
//        let accuracy = Float(correct) / Float(predictionProvider!.count)
//
//        print("Accuracy: \(accuracy)")
//        self.accuracy = "Accuracy: \(accuracy)"
//    }
//
//    public func predict(data: [[Float]]) -> Int {
//        let imageMultiArr = try! MLMultiArray(shape: [1, 28, 28], dataType: .float32)
//
//        for r in 0..<28 {
//            for c in 0..<28 {
//                let i = (r*28)+c
//                imageMultiArr[i] = NSNumber(value: data[r][c]) // already normalized
//            }
//        }
//
//        let imageValue = MLFeatureValue(multiArray: imageMultiArr)
//
//        let dataPointFeatures: [String: MLFeatureValue] = ["image": imageValue]
//
//        let provider = try! MLDictionaryFeatureProvider(dictionary: dataPointFeatures)
//
//        guard let prediction = try! retrainedModel?.prediction(from: provider) else { return -1 }
//
//        let oneHotPrediction = prediction.featureValue(for: "output")!
//
//        var predictedNumber = -1
//        var max: Float = -1.0
//
//        for i in 0..<10 {
//            if oneHotPrediction.multiArrayValue![i].floatValue > max {
//                predictedNumber = i
//                max = oneHotPrediction.multiArrayValue![i].floatValue
//            }
//        }
//
//        return predictedNumber
//    }
}

