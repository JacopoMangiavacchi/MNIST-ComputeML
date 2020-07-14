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
    @Published public var testBatchCount = 0
    @Published public var trainingDataX: [Float]?
    @Published public var trainingDataY: [Float]?
    @Published public var testDataX: [Float]?
    @Published public var testDataY: [Float]?
    @Published public var dataPreparing = false
    @Published public var modelTraining = false
    @Published public var modelTrained = false
    @Published public var trainingFeedback = "Train the model"
    @Published public var epochs: Int = 5
    
    let batchSize = 32
    let dense1LayerOutputSize = 128

    var device: MLCDevice!
    var graph: MLCGraph!
    var trainingGraph: MLCTrainingGraph!
    var inferenceGraph: MLCInferenceGraph!
    var inputTensor: MLCTensor!
    var dense1WeightsTensor: MLCTensor!
    var dense1BiasesTensor: MLCTensor!
    var dense2WeightsTensor: MLCTensor!
    var dense2BiasesTensor: MLCTensor!
    var dense1: MLCTensor!
    var relu1: MLCTensor!
    var dense2: MLCTensor!
    var outputSoftmax: MLCTensor!
    var lossLabelTensor: MLCTensor!

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
    
    private func oneHotEncoding(_ number: Int, length: Int = 10) -> [Float] {
        guard number < length else {
            fatalError("wrong ordinal vs encoding length")
        }
        
        var array = Array<Float>(repeating: 0.0, count: length)
        array[number] = 1.0
        return array
    }
    
    private func oneHotDecoding(_ encoding: [Float]) -> Int {
        var value: Int = 0
        
        for i in 0..<encoding.count {
            if encoding[i] == 1 {
                value = i
                break
            }
        }
        
        return value
    }
    
    private func argmaxDecoding(_ encoding: [Float]) -> Int {
        var max: Float = 0
        var pos: Int = 0
        
        for i in 0..<encoding.count {
            if encoding[i] > max {
                max = encoding[i]
                pos = i
            }
        }
        
        return pos
    }
    
    private func readDataSet(fileName: String, updateStatus: @escaping (Int) -> Void) -> ([Float], [Float]) { //}(MLCTensor, MLCTensor) {
        guard let filePath = Bundle.main.path(forResource: fileName, ofType: "csv") else {
            fatalError("CSV file not found")
        }

        let serialQueue = DispatchQueue(label: "MNIST.serial.queue.\(fileName)")
        
        var count = 0
        var X = [Float]()
        var Y = [Float]()
        
        let iterations = 20
        var iteration = 0
        var iterationList = Array<Array<String>>(repeating: Array<String>(), count: iterations)

        getFileLine(filePath: filePath) { line in
            iterationList[iteration].append(line)
            iteration = (iteration + 1) % iterations
        }
        
        DispatchQueue.concurrentPerform(iterations: iterations) { iteration in
            for line in iterationList[iteration] {
                let sample = line.split(separator: ",").compactMap({Int($0)})

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
        
    public func asyncPrepareData() {
        trainingBatchCount = 0
        testBatchCount = 0
        dataPreparing = true
        var trainPrepared = false
        var testPrepared = false

        concurrentQueue.async {
            let (X, Y) = self.readDataSet(fileName: "mnist_train") { count in
                DispatchQueue.main.async {
                    self.trainingBatchCount = count
                }
            }
            
            DispatchQueue.main.async {
                self.trainingBatchCount = X.count / self.imageSize
                self.trainingDataX = X
                self.trainingDataY = Y
                
                trainPrepared = true
                if testPrepared {
                    self.dataPreparing = false
                }
            }
        }

        concurrentQueue.async {
            let (X, Y) = self.readDataSet(fileName: "mnist_test") { count in
                DispatchQueue.main.async {
                    self.testBatchCount = count
                }
            }
            
            DispatchQueue.main.async {
                self.testBatchCount = X.count / self.imageSize
                self.testDataX = X
                self.testDataY = Y

                testPrepared = true
                if trainPrepared {
                    self.dataPreparing = false
                }
            }
        }
    }
    
    private func initializeTensors() {
        device = MLCDevice(type: .cpu)!
        
        inputTensor = MLCTensor(descriptor: MLCTensorDescriptor(shape: [batchSize, imageSize, 1, 1], dataType: .float32)!)
        
        dense1WeightsTensor = MLCTensor(descriptor: MLCTensorDescriptor(shape: [1, imageSize*dense1LayerOutputSize, 1, 1], dataType: .float32)!,
                                            randomInitializerType: .glorotUniform)
        dense1BiasesTensor = MLCTensor(descriptor: MLCTensorDescriptor(shape: [1, dense1LayerOutputSize, 1, 1], dataType: .float32)!,
                                           randomInitializerType: .glorotUniform)
        dense2WeightsTensor = MLCTensor(descriptor: MLCTensorDescriptor(shape: [1, dense1LayerOutputSize*numberOfClasses, 1, 1], dataType: .float32)!,
                                            randomInitializerType: .glorotUniform)
        dense2BiasesTensor = MLCTensor(descriptor: MLCTensorDescriptor(shape: [1, numberOfClasses, 1, 1], dataType: .float32)!,
                                           randomInitializerType: .glorotUniform)

        lossLabelTensor = MLCTensor(descriptor: MLCTensorDescriptor(shape: [batchSize, numberOfClasses], dataType: .float32)!)
    }
    
    private func buildGraph() {
        graph = MLCGraph()
        
        dense1 = graph.node(with: MLCFullyConnectedLayer(weights: dense1WeightsTensor,
                                                            biases: dense1BiasesTensor,
                                                            descriptor: MLCConvolutionDescriptor(kernelSizes: (height: imageSize, width: dense1LayerOutputSize),
                                                                                                 inputFeatureChannelCount: imageSize,
                                                                                                 outputFeatureChannelCount: dense1LayerOutputSize))!,
                               sources: [inputTensor])
        
        relu1 = graph.node(with: MLCActivationLayer(descriptor: MLCActivationDescriptor(type: MLCActivationType.relu)!),
                   source: dense1!)

        dense2 = graph.node(with: MLCFullyConnectedLayer(weights: dense2WeightsTensor,
                                                            biases: dense2BiasesTensor,
                                                            descriptor: MLCConvolutionDescriptor(kernelSizes: (height: dense1LayerOutputSize, width: numberOfClasses),
                                                                                                 inputFeatureChannelCount: dense1LayerOutputSize,
                                                                                                 outputFeatureChannelCount: numberOfClasses))!,
                               sources: [relu1!])

        outputSoftmax = graph.node(with: MLCSoftmaxLayer(operation: .softmax),
                   source: dense2!)
    }
    
    private func buildTrainingGraph() {
        trainingGraph = MLCTrainingGraph(graphObjects: [graph],
                                             lossLayer: MLCLossLayer(descriptor: MLCLossDescriptor(type: .softmaxCrossEntropy,
                                                                                                   reductionType: .mean)),
                                             optimizer: MLCAdamOptimizer(descriptor: MLCOptimizerDescriptor(learningRate: 0.001,
                                                                                                           gradientRescale: 1.0,
                                                                                                        regularizationType: .none,
                                                                                                        regularizationScale: 0.0),
                                                                         beta1: 0.9,
                                                                         beta2: 0.999,
                                                                         epsilon: 1e-7,
                                                                         timeStep: 1))

        trainingGraph.addInputs(["image" : inputTensor],
                                lossLabels: ["label" : lossLabelTensor])
        
        trainingGraph.compile(options: [], device: device)
    }
    
    private func execTrainingLoop(log: (String) -> Void) {
        let trainingSample = trainingDataX!.count / imageSize
        let trainingBatches = trainingSample / batchSize

        for epoch in 0..<epochs {
            var epochMatch = 0

            for batch in 0..<trainingBatches {
                let xData = trainingDataX!.withUnsafeBufferPointer { pointer in
                    MLCTensorData(immutableBytesNoCopy: pointer.baseAddress!.advanced(by: batch * imageSize * batchSize),
                                  length: batchSize * imageSize * MemoryLayout<Float>.size)
                }

                let yData = trainingDataY!.withUnsafeBufferPointer { pointer in
                    MLCTensorData(immutableBytesNoCopy: pointer.baseAddress!.advanced(by: batch * numberOfClasses * batchSize),
                                  length: batchSize * numberOfClasses * MemoryLayout<Int>.size)
                }
                
                trainingGraph.execute(inputsData: ["image" : xData],
                                      lossLabelsData: ["label" : yData],
                                      lossLabelWeightsData: nil,
                                      batchSize: batchSize,
                                      options: [.synchronous]) { [self] (r, e, time) in
                    // VALIDATE
                    let bufferOutput = UnsafeMutableRawPointer.allocate(byteCount: batchSize * self.numberOfClasses * MemoryLayout<Float>.size, alignment: MemoryLayout<Float>.alignment)

//                    r!.copyDataFromDeviceMemory(toBytes: bufferOutput, length: batchSize * self.numberOfClasses * MemoryLayout<Float>.size, synchronizeWithDevice: false)
                    outputSoftmax!.copyDataFromDeviceMemory(toBytes: bufferOutput, length: batchSize * self.numberOfClasses * MemoryLayout<Float>.size, synchronizeWithDevice: false)

                    let float4Ptr = bufferOutput.bindMemory(to: Float.self, capacity: batchSize * self.numberOfClasses)
                    let float4Buffer = UnsafeBufferPointer(start: float4Ptr, count: batchSize * self.numberOfClasses)
                    let batchOutputArray = Array(float4Buffer)
                    
//                    let batchOutputArray = outputTensor.data!.withUnsafeBytes { (bytes: UnsafePointer<Float>) in
//                        Array(UnsafeBufferPointer(start: bytes, count: batchSize * self.numberOfClasses))
//                    }

                    for i in 0..<batchSize {
                        let batchStartingPoint = i * self.numberOfClasses
                        let predictionStartingPoint = (i * self.numberOfClasses) + (batch * batchSize * numberOfClasses)
                        let sampleOutputArray = Array(batchOutputArray[batchStartingPoint..<(batchStartingPoint + self.numberOfClasses)])
                        let predictionArray = Array(trainingDataY![predictionStartingPoint..<(predictionStartingPoint + numberOfClasses)])
                        
                        let prediction = argmaxDecoding(sampleOutputArray)
                        let label = oneHotDecoding(predictionArray)
                        
                        if prediction == label {
                            epochMatch += 1
                        }
                        
                        // print("\(i + (batch * batchSize)) -> Prediction: \(prediction) Label: \(label)")
                    }
                }
            }
            
            let epochAccuracy = Float(epochMatch) / Float(trainingSample)
            log("Epoch \(epoch) Accuracy = \(epochAccuracy) %")
        }
    }
    
    private func evaluateGraph(log: (String) -> Void) {
        let testingSample = testDataX!.count / imageSize
        let testingBatches = testingSample / batchSize

        
        inferenceGraph = MLCInferenceGraph(graphObjects: [graph])
        inferenceGraph.addInputs(["image" : inputTensor])
        inferenceGraph.compile(options: [], device: device)

        // TESTING LOOP FOR A FULL EPOCH ON TESTING DATA
        var match = 0
        
        for batch in 0..<testingBatches {
            let xData = testDataX!.withUnsafeBufferPointer { pointer in
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
                    let predictionArray = Array(testDataY![predictionStartingPoint..<(predictionStartingPoint + numberOfClasses)])
                    
                    let prediction = argmaxDecoding(sampleOutputArray)
                    let label = oneHotDecoding(predictionArray)
                    
                    if prediction == label {
                        match += 1
                    }
                    
                    // print("\(i + (batch * batchSize)) -> Prediction: \(prediction) Label: \(label)")
                }
            }
        }
        
        let accuracy = Float(match) / Float(testingSample)
        log("Test Accuracy = \(accuracy) %")
    }
    
    private func trainGraph(log: (String) -> Void) {
        // MODEL
        // -----
        // model = keras.Sequential([
        //     keras.layers.Dense(128, activation='relu'),  // W (784, 128)  B (128,)
        //     keras.layers.Dense(10)                       // W (128, 10)   B (10,)
        // ])

        initializeTensors()
        
        buildGraph()
        
        buildTrainingGraph()
        
        execTrainingLoop(log: log)

        evaluateGraph(log: log)
    }
    
    public func asyncTrainGraph() {
        modelTraining = true
        modelTrained = false
        trainingFeedback = "Training.."
        
        concurrentQueue.async {
            self.trainGraph{ logText in
                print(logText)
                DispatchQueue.main.async {
                    self.trainingFeedback = logText
                }
            }
            
            DispatchQueue.main.async {
                self.modelTraining = false
                self.modelTrained = true
            }
        }
    }
    
    public func predict(data: [[Float]]) -> Int {
        var image: [Float] = Array(data.joined())
        image.append(contentsOf: Array<Float>(repeating: 0.0, count: (batchSize - 1) * imageSize))
        
        
        let xData = image.withUnsafeBufferPointer { pointer in
            MLCTensorData(immutableBytesNoCopy: pointer.baseAddress!,
                          length: batchSize * imageSize * MemoryLayout<Float>.size)
        }
        
        var prediction = -1
        inferenceGraph.execute(inputsData: ["image" : xData],
                              batchSize: batchSize,
                              options: [.synchronous]) { [self] (r, e, time) in
            let bufferOutput = UnsafeMutableRawPointer.allocate(byteCount: batchSize * numberOfClasses * MemoryLayout<Float>.size, alignment: MemoryLayout<Float>.alignment)

            r!.copyDataFromDeviceMemory(toBytes: bufferOutput, length: batchSize * numberOfClasses * MemoryLayout<Float>.size, synchronizeWithDevice: false)

            let float4Ptr = bufferOutput.bindMemory(to: Float.self, capacity: batchSize * numberOfClasses)
            let float4Buffer = UnsafeBufferPointer(start: float4Ptr, count: batchSize * numberOfClasses)
            let batchOutputArray = Array(float4Buffer)
            let firstImageOutput = Array(batchOutputArray[0..<numberOfClasses])

            prediction = argmaxDecoding(firstImageOutput)

            print(prediction)
        }
        
        return prediction
    }
}
