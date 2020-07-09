//
//  MNIST.swift
//  MNIST-ComputeML
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
    @Published public var trainingBatchProviderX: [Float]?
    @Published public var trainingBatchProviderY: [Int64]?
    @Published public var predictionBatchProviderX: [Float]?
    @Published public var predictionBatchProviderY: [Int64]?
    @Published public var modelPrepared = false
    @Published public var modelCompiled = false
    @Published public var modelTrained = false
    @Published public var modelStatus = "Train model"
    @Published public var accuracy = "Accuracy: n/a"
    @Published public var epoch: Int = 5
    
    // Load in memory and split is not performant
    private func getFileLine(filePath: String, process: (String) -> Void) {
        guard let filePointer:UnsafeMutablePointer<FILE> = fopen(filePath,"r") else {
            preconditionFailure("Could not open file at \(filePath)")
        }

        defer {
            fclose(filePointer)
        }

        var lineByteArrayPointer: UnsafeMutablePointer<CChar>? = nil
        var lineCap: Int = 0

        while getline(&lineByteArrayPointer, &lineCap, filePointer) > 0 {
            let line = String.init(cString:lineByteArrayPointer!).trimmingCharacters(in: .whitespacesAndNewlines)

            process(line)
        }
    }
    
    public func readDataSet(fileName: String, updateStatus: @escaping (Int) -> Void) -> ([Float], [Int64]) { //}(MLCTensor, MLCTensor) {
        guard let filePath = Bundle.main.path(forResource: fileName, ofType: "csv") else {
            fatalError("CSV file not found")
        }

        let serialQueue = DispatchQueue(label: "MNIST.serial.queue.\(fileName)")
        
        var count = 0
        var X = [Float]()
        var Y = [Int64]()
        
        let iterations = 20
        var iteration = 0
        var iterationList = Array<Array<String>>(repeating: Array<String>(), count: iterations)

        getFileLine(filePath: filePath) { line in
            iterationList[iteration].append(line)
            iteration = (iteration + 1) % iterations
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
        
        return (X, Y)
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
                self.trainingBatchCount = X.count / self.imageSize
                self.trainingBatchProviderX = X
                self.trainingBatchProviderY = Y
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
                self.predictionBatchCount = X.count / self.imageSize
                self.predictionBatchProviderX = X
                self.predictionBatchProviderY = Y
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
        let dense1 = graph.node(with: MLCFullyConnectedLayer(weights: MLCTensor(descriptor: MLCTensorDescriptor(shape: [1, imageSize*128, 1, 1], dataType: .float32)!,
                                                                                randomInitializerType: .glorotUniform),
                                                            biases: MLCTensor(descriptor: MLCTensorDescriptor(shape: [1, 128, 1, 1], dataType: .float32)!,
                                                                              randomInitializerType: .glorotUniform),
                                                            descriptor: MLCConvolutionDescriptor(kernelSizes: (height: imageSize, width: 128),
                                                                                                 inputFeatureChannelCount: imageSize,
                                                                                                 outputFeatureChannelCount: 128))!,
                               sources: [MLCTensor(descriptor: MLCTensorDescriptor(shape: [1, imageSize, 1, 1], dataType: .float32)!)])
        
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
        graph.node(with: MLCSoftmaxLayer(operation: .softmax),
                   source: dense2!)
        
        let trainingGraph = MLCTrainingGraph(graphObjects: [graph],
                                             lossLayer: MLCLossLayer(descriptor: MLCLossDescriptor(type: .categoricalCrossEntropy,
                                                                                                   reductionType: .none)),
                                             optimizer: MLCSGDOptimizer(descriptor: MLCOptimizerDescriptor(learningRate: 0.1,
                                                                                                        gradientRescale: 0.1,
                                                                                                        regularizationType: .none,
                                                                                                        regularizationScale: 0.0)))

        trainingGraph.addInputs(["image" : MLCTensor(descriptor: MLCTensorDescriptor(shape: [imageSize, 1], dataType: .float32)!)],
                                lossLabels: ["label" : MLCTensor(descriptor: MLCTensorDescriptor(shape: [10, 1], dataType: .int64)!)])

        let device = MLCDevice(type: .cpu)!
        
        let b = trainingGraph.compile(options: [], device: device)
        print(b)
        
        let xData = trainingBatchProviderX!.withUnsafeBufferPointer { pointer in
            MLCTensorData(immutableBytesNoCopy: pointer.baseAddress!,
                          length: pointer.count * MemoryLayout<Float>.size)
        }

        let yData = trainingBatchProviderY!.withUnsafeBufferPointer { pointer in
            MLCTensorData(immutableBytesNoCopy: pointer.baseAddress!,
                          length: pointer.count * MemoryLayout<Int>.size)
        }

//        let count = trainingBatchProviderX!.count / imageSize
//
//        let xTensor = MLCTensor(descriptor: MLCTensorDescriptor(shape: [count, imageSize], dataType: .float32)!,
//                                data: xData)
//
//        let yTensor = MLCTensor(descriptor: MLCTensorDescriptor(shape: [count, 1], dataType: .int64)!,
//                                data: yData)
        

        
        trainingGraph.execute(inputsData: ["image" : xData],
                              lossLabelsData: ["label" : yData],
                              lossLabelWeightsData: nil,
                              batchSize: 1,
                              options: []) { (r, e, time) in
            print("Error: \(String(describing: e))")
            print("Result: \(String(describing: r))")

//            let buffer3 = UnsafeMutableRawPointer.allocate(byteCount: 6 * MemoryLayout<Float>.size, alignment: MemoryLayout<Float>.alignment)
//
//            r!.copyDataFromDeviceMemory(toBytes: buffer3, length: 6 * MemoryLayout<Float>.size, synchronizeWithDevice: false)
//
//            let float4Ptr = buffer3.bindMemory(to: Float.self, capacity: 6)
//            let float4Buffer = UnsafeBufferPointer(start: float4Ptr, count: 6)
//            print(Array(float4Buffer))

        }
    }
    
}

