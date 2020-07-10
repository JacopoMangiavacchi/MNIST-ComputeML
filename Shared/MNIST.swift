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
    let numberOfClasses = 10
    let expextedTrainingSamples = 60000
    let expectedTestingSamples = 10000

    let concurrentQueue = DispatchQueue(label: "MNIST.concurrent.queue", attributes: .concurrent)

    @Published public var trainingBatchCount = 0
    @Published public var predictionBatchCount = 0
    @Published public var trainingBatchProviderX: [Float]?
    @Published public var trainingBatchProviderY: [Int64]?
    @Published public var predictionBatchProviderX: [Float]?
    @Published public var predictionBatchProviderY: [Int64]?
    @Published public var modelTrained = false
    @Published public var epochs: Int = 5
    
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
    
    func oneHotEncoding(_ number: Int64, length: Int = 10) -> [Int64] {
        guard number < length else {
            fatalError("wrong ordinal vs encoding length")
        }
        
        var array = Array<Int64>(repeating: 0, count: length)
        array[Int(number)] = 1
        return array
    }
    
    func decode(_ encoding: [Int64], length: Int = 10) -> Int64 {
        var value: Int64 = 0
        
        for i in 0..<length {
            if encoding[i] == 1 {
                value = Int64(i)
                break
            }
        }
        
        return value
    }
    
    func argmaxDecode(_ encoding: [Float], length: Int = 10) -> Int64 {
        var max: Float = 0
        var pos: Int64 = 0
        
        for i in 0..<length {
            if encoding[i] > max {
                max = encoding[i]
                pos = Int64(i)
            }
        }
        
        return pos
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
                    Y.append(contentsOf: oneHotEncoding(sample[0]))
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
    
    public func trainGraph() {
        // MODEL
        // -----
        // model = keras.Sequential([
        //     keras.layers.Dense(128, activation='relu'),  // W (784, 128)  B (128,)
        //     keras.layers.Dense(10)                       // W (128, 10)   B (10,)
        // ])
        
        let batchSize = 25
        let trainingSample = trainingBatchProviderX!.count / imageSize
        let testingSample = predictionBatchProviderX!.count / imageSize
        let trainBatches = trainingSample / batchSize
        let testingBatches = testingSample / batchSize

        let dense1LayerOutputSize = 128

        let device = MLCDevice(type: .cpu)!

        let inputTensor = MLCTensor(descriptor: MLCTensorDescriptor(shape: [batchSize, imageSize, 1, 1], dataType: .float32)!)
        let lossLabelTensor = MLCTensor(descriptor: MLCTensorDescriptor(shape: [batchSize, numberOfClasses], dataType: .int64)!)
        
        let dense1WeightsTensor = MLCTensor(descriptor: MLCTensorDescriptor(shape: [1, imageSize*dense1LayerOutputSize, 1, 1], dataType: .float32)!,
                                            randomInitializerType: .glorotUniform)
        let dense1BiasesTensor = MLCTensor(descriptor: MLCTensorDescriptor(shape: [1, dense1LayerOutputSize, 1, 1], dataType: .float32)!,
                                           randomInitializerType: .glorotUniform)
        let dense2WeightsTensor = MLCTensor(descriptor: MLCTensorDescriptor(shape: [1, dense1LayerOutputSize*numberOfClasses, 1, 1], dataType: .float32)!,
                                            randomInitializerType: .glorotUniform)
        let dense2BiasesTensor = MLCTensor(descriptor: MLCTensorDescriptor(shape: [1, numberOfClasses, 1, 1], dataType: .float32)!,
                                           randomInitializerType: .glorotUniform)

        // CREATE TRAINING GRAPH

        let graph = MLCGraph()

        let dense1 = graph.node(with: MLCFullyConnectedLayer(weights: dense1WeightsTensor,
                                                            biases: dense1BiasesTensor,
                                                            descriptor: MLCConvolutionDescriptor(kernelSizes: (height: imageSize, width: dense1LayerOutputSize),
                                                                                                 inputFeatureChannelCount: imageSize,
                                                                                                 outputFeatureChannelCount: dense1LayerOutputSize))!,
                               sources: [inputTensor])
        
        let dense2 = graph.node(with: MLCFullyConnectedLayer(weights: dense2WeightsTensor,
                                                            biases: dense2BiasesTensor,
                                                            descriptor: MLCConvolutionDescriptor(kernelSizes: (height: dense1LayerOutputSize, width: numberOfClasses),
                                                                                                 inputFeatureChannelCount: dense1LayerOutputSize,
                                                                                                 outputFeatureChannelCount: numberOfClasses))!,
                               sources: [dense1!])

        graph.node(with: MLCSoftmaxLayer(operation: .softmax),
                   source: dense2!)
        
        let trainingGraph = MLCTrainingGraph(graphObjects: [graph],
                                             lossLayer: MLCLossLayer(descriptor: MLCLossDescriptor(type: .categoricalCrossEntropy,
                                                                                                   reductionType: .none)),
                                             optimizer: MLCSGDOptimizer(descriptor: MLCOptimizerDescriptor(learningRate: 0.001,
                                                                                                           gradientRescale: 0.5,
                                                                                                        regularizationType: .none,
                                                                                                        regularizationScale: 0.0)))

        trainingGraph.addInputs(["image" : inputTensor],
                                lossLabels: ["label" : lossLabelTensor])

        
        trainingGraph.compile(options: [], device: device)
        
        // TRAINING LOOP
        for epoch in 0..<epochs {
            for batch in 0..<trainBatches {
                let xData = trainingBatchProviderX!.withUnsafeBufferPointer { pointer in
                    MLCTensorData(immutableBytesNoCopy: pointer.baseAddress!.advanced(by: batch * imageSize * batchSize),
                                  length: batchSize * imageSize * MemoryLayout<Float>.size)
                }

                let yData = trainingBatchProviderY!.withUnsafeBufferPointer { pointer in
                    MLCTensorData(immutableBytesNoCopy: pointer.baseAddress!.advanced(by: batch * numberOfClasses * batchSize),
                                  length: batchSize * numberOfClasses * MemoryLayout<Int>.size)
                }
                
                trainingGraph.execute(inputsData: ["image" : xData],
                                      lossLabelsData: ["label" : yData],
                                      lossLabelWeightsData: nil,
                                      batchSize: batchSize,
                                      options: [.synchronous]) { (r, e, time) in
                    // TODO: VALIDATE !!
                }
            }
        }
        
        // CREATE INFERENCE GRAPH REUSING TRAINING WEIGHTS/BIASES
        let inferenceGraph = MLCInferenceGraph(graphObjects: [graph])
        inferenceGraph.addInputs(["image" : inputTensor])
        inferenceGraph.compile(options: [], device: device)

        // TESTING LOOP FOR A FULL EPOCH ON TESTING DATA
        for batch in 0..<testingBatches {
            let xData = predictionBatchProviderX!.withUnsafeBufferPointer { pointer in
                MLCTensorData(immutableBytesNoCopy: pointer.baseAddress!.advanced(by: batch * imageSize * batchSize),
                              length: batchSize * imageSize * MemoryLayout<Float>.size)
            }
            
            inferenceGraph.execute(inputsData: ["image" : xData],
                                  batchSize: batchSize,
                                  options: [.synchronous]) { [self] (r, e, time) in
//                print("Batch \(batch) Error: \(String(describing: e))")

                let bufferOutput = UnsafeMutableRawPointer.allocate(byteCount: batchSize * numberOfClasses * MemoryLayout<Float>.size, alignment: MemoryLayout<Float>.alignment)

                r!.copyDataFromDeviceMemory(toBytes: bufferOutput, length: batchSize * numberOfClasses * MemoryLayout<Float>.size, synchronizeWithDevice: false)

                let float4Ptr = bufferOutput.bindMemory(to: Float.self, capacity: batchSize * numberOfClasses)
                let float4Buffer = UnsafeBufferPointer(start: float4Ptr, count: batchSize * numberOfClasses)
                let batchOutputArray = Array(float4Buffer)

                for i in 0..<batchSize {
                    let batchStartingPoint = i * numberOfClasses
                    let predictionStartingPoint = (i * numberOfClasses) + (batch * batchSize * numberOfClasses)
                    let sampleOutputArray = Array(batchOutputArray[batchStartingPoint..<(batchStartingPoint + numberOfClasses)])
                    let predictionArray = Array(predictionBatchProviderY![predictionStartingPoint..<(predictionStartingPoint + numberOfClasses)])
                    print("\(i + (batch * batchSize)) -> Esitmate: \(argmaxDecode(sampleOutputArray)) Label: \(decode(predictionArray))")
                }
            }
        }
        
        modelTrained = true
    }
}
