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

    @Published public var trainingBatchCount = 0
    @Published public var predictionBatchCount = 0
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
    
    public func readDataSet(fileName: String, updateStatus: @escaping (Int) -> Void) -> (MLCTensor, MLCTensor) {
        let serialQueue = DispatchQueue(label: "MNIST.serial.queue.\(fileName)")
        
        var count = 0
        var X = [Float]()
        var Y = [Int64]()

        guard let filePath = Bundle.main.path(forResource: fileName, ofType: "csv") else {
            fatalError("CSV file not found")
        }
        guard let filePointer:UnsafeMutablePointer<FILE> = fopen(filePath,"r") else {
            preconditionFailure("Could not open file at \(filePath)")
        }

        var lineByteArrayPointer: UnsafeMutablePointer<CChar>? = nil
        var lineCap: Int = 0
        var bytesRead = getline(&lineByteArrayPointer, &lineCap, filePointer)

        defer {
            fclose(filePointer)
        }

        let iterations = 20
        var iteration = 0
        var iterationList = Array<Array<String>>(repeating: Array<String>(), count: iterations)
        
        while (bytesRead > 0) {
            let line = String.init(cString:lineByteArrayPointer!).trimmingCharacters(in: .whitespacesAndNewlines)

            iterationList[iteration].append(line)
            iteration = (iteration + 1) % iterations
            
            bytesRead = getline(&lineByteArrayPointer, &lineCap, filePointer)
        }
        
        DispatchQueue.concurrentPerform(iterations: iterations) { iteration in
            for line in iterationList[iteration] {
                let sample = line.split(separator: ",").compactMap({Int64($0)})

                serialQueue.sync {
                    Y.append(sample[0])
                    X.append(contentsOf: sample[1...self.imageSize].map{Float($0) / Float(255.0)})
                    
                    count += 1
                    updateStatus(count)
                }
            }
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
        self.trainingBatchCount = 0
        concurrentQueue.async {
            let (X, Y) = self.readDataSet(fileName: "mnist_train") { count in
                DispatchQueue.main.async {
                    self.trainingBatchCount = count
                }
            }
            
            DispatchQueue.main.async {
                self.trainingBatchCount = X.descriptor.shape[0]
                self.trainingBatchProviderXTensor = X
                self.trainingBatchProviderYTensor = Y
            }
        }
    }
    
    public func asyncPreparePredictionBatchProvider() {
        self.predictionBatchCount = 0
        concurrentQueue.async {
            let (X, Y) = self.readDataSet(fileName: "mnist_test") { count in
                DispatchQueue.main.async {
                    self.predictionBatchCount = count
                }
            }
            
            DispatchQueue.main.async {
                self.predictionBatchCount = X.descriptor.shape[0]
                self.predictionBatchProviderXTensor = X
                self.predictionBatchProviderYTensor = Y
            }
        }
    }
    
    public func prepareGraph() {
        // MODEL
        // -----
        // model = keras.Sequential([
        //     keras.layers.Dense(128, activation='relu'),  // W (784, 128)  B (128,)
        //     keras.layers.Dense(10)                       // W (128, 10)   B (10,)
        // ])

        let graph = MLCGraph()
        
        // DENSE LAYER
        // -----------
        //  INPUT SHAPE: (784, 1)
        //  LABEL SHAPE: (10, 1)
        //  OUTPUT SHAPE: (128, 1)
        //  NB Weights and Bias have to be 4d shaped
        let dense1 = graph.node(with: MLCFullyConnectedLayer(weights: MLCTensor(descriptor: MLCTensorDescriptor(shape: [1, 784*128, 1, 1], dataType: .float32)!,
                                                                                randomInitializerType: .glorotUniform),
                                                            biases: MLCTensor(descriptor: MLCTensorDescriptor(shape: [1, 128, 1, 1], dataType: .float32)!,
                                                                              randomInitializerType: .glorotUniform),
                                                            descriptor: MLCConvolutionDescriptor(kernelSizes: (height: 784, width: 128),
                                                                                                 inputFeatureChannelCount: 784,
                                                                                                 outputFeatureChannelCount: 128))!,
                               sources: [MLCTensor(descriptor: MLCTensorDescriptor(shape: [784, 1], dataType: .float32)!)])
        
        // DENSE LAYER
        // -----------
        //  INPUT SHAPE: (128, 1)
        //  OUTPUT SHAPE: (10, 1)
        let dense2 = graph.node(with: MLCFullyConnectedLayer(weights: MLCTensor(descriptor: MLCTensorDescriptor(shape: [1, 128*10, 1, 1], dataType: .float32)!,
                                                                                randomInitializerType: .glorotUniform),
                                                            biases: MLCTensor(descriptor: MLCTensorDescriptor(shape: [1, 10, 1, 1], dataType: .float32)!,
                                                                              randomInitializerType: .glorotUniform),
                                                            descriptor: MLCConvolutionDescriptor(kernelSizes: (height: 128, width: 10),
                                                                                                 inputFeatureChannelCount: 128,
                                                                                                 outputFeatureChannelCount: 10))!,
                               sources: [dense1!])
        
        // SOFTMAX ACTIVATION
        // ------------------
        graph.node(with: MLCSoftmaxLayer(operation: MLCSoftmaxOperation(rawValue: 10)!),
                   source: dense2!)
        
        let trainingGraph = MLCTrainingGraph(graphObjects: [graph],
                                             lossLayer: MLCLossLayer(descriptor: MLCLossDescriptor(type: .softmaxCrossEntropy,
                                                                                                   reductionType: .none)),
                                             optimizer: MLCOptimizer(descriptor: MLCOptimizerDescriptor(learningRate: 0.1,
                                                                                                        gradientRescale: 0.1,
                                                                                                        regularizationType: .none,
                                                                                                        regularizationScale: 0.0)))
        
        trainingGraph.addInputs(["image" : MLCTensor(descriptor: MLCTensorDescriptor(shape: [784, 1], dataType: .float32)!)],
                                lossLabels: ["label" : MLCTensor(descriptor: MLCTensorDescriptor(shape: [10, 1], dataType: .int64)!)])

//        print(trainingGraph)

        let b = trainingGraph.compile(options: [], device: MLCDevice(type: .cpu)!)
        
        print(b)
        

//        let i = MLCInferenceGraph(graphObjects: [g])
//        i.addInputs(["data1" : tensor1, "data2" : tensor2, "data3" : tensor3])
//        i.compile(options: .debugLayers, device: MLCDevice())
//
//        i.execute(inputsData: ["data1" : data1, "data2" : data2, "data3" : data3],
//                  batchSize: 0,
//                  options: []) { (r, e, time) in
//            print("Error: \(String(describing: e))")
//            print("Result: \(String(describing: r))")
//
//            let buffer3 = UnsafeMutableRawPointer.allocate(byteCount: 6 * MemoryLayout<Float>.size, alignment: MemoryLayout<Float>.alignment)
//
//            r!.copyDataFromDeviceMemory(toBytes: buffer3, length: 6 * MemoryLayout<Float>.size, synchronizeWithDevice: false)
//
//            let float4Ptr = buffer3.bindMemory(to: Float.self, capacity: 6)
//            let float4Buffer = UnsafeBufferPointer(start: float4Ptr, count: 6)
//            print(Array(float4Buffer))
//
//            page.finishExecution()
//        }

        

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

