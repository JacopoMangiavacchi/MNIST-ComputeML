//
//  ContentView.swift
//  Shared
//
//  Created by Jacopo Mangiavacchi on 6/28/20.
//

import SwiftUI

import SwiftUI

struct ContentView: View {
    @ObservedObject var mnist = MNIST()
    @ObservedObject var drawData = DrawData()
    @State var prediction = "-"
    
    func isDataReady(trainingCount: Int, predictionCount: Int) -> Bool {
        return trainingCount == 60000 && predictionCount == 10000
    }
    
    let splitRatio: CGFloat = 0.2445
    
    var body: some View {
        GeometryReader { geometry in
            VStack(spacing: 0) {
                Form {
                    Section(header: Text("Dataset")) {
                        HStack {
                            ProgressView("Training: \(self.mnist.trainingBatchCount)", value: Float(self.mnist.trainingBatchCount), total: 60000)
                            Spacer(minLength: 50.0)
                            ProgressView("Test: \(self.mnist.predictionBatchCount)", value: Float(self.mnist.predictionBatchCount), total: 10000)
                            Spacer(minLength: 100.0)
                            Button(action: {
                                self.mnist.asyncPrepareTrainBatchProvider()
                                self.mnist.asyncPreparePredictionBatchProvider()
                            }) {
                                Text("Start")
                            }
                        }
                    }
                    Section(header: Text("Training")) {
                        Stepper(value: self.$mnist.epoch, in: 1...10, label: { Text("Epoch:  \(self.mnist.epoch)")})
                        HStack {
                            Text("Compile the model")
                            Spacer()
                            Button(action: {
                                self.mnist.prepareGraph()
                            }) {
                                Text("Start")
                            }//.disabled(!self.isDataReady(trainingCount: self.mnist.trainingBatchCount, predictionCount: self.mnist.predictionBatchCount))
                        }
//                        HStack {
//                            Text(self.mnist.modelStatus)
//                            Spacer()
//                            Button(action: {
//                                self.mnist.trainModel()
//                            }) {
//                                Text("Start")
//                            }.disabled(!self.mnist.modelCompiled)
//                        }
                    }
//                    Section(header: Text("Validation")) {
//                        HStack {
//                            Text("Predict Test data")
//                            Spacer()
//                            Button(action: {
//                                self.mnist.testModel()
//                            }) {
//                                Text("Start")
//                            }.disabled(!self.isDataReady(for: self.mnist.predictionBatchStatus) || !self.mnist.modelTrained)
//                        }
//                        Text(self.mnist.accuracy)
//                    }
                    Section(header: Text("Test")) {
                        HStack {
                            Button(action: {}) {
                                Text("Clear")
                            }.onTapGesture {
                                self.prediction = "-"
                                self.drawData.lines.removeAll()
                            }
                            Spacer()
                            Text(self.prediction)
                            Spacer()
                            Button(action: {}) {
                                Text("Predict")
                            }
                                .disabled(!self.mnist.modelTrained)
                                .onTapGesture {
                                    let data = self.drawData.view.getImageData()
                                    // self.prediction = "\(self.mnist.predict(data: data))"
                                }
                        }
                    }
                }.frame(width: geometry.size.width, height: geometry.size.height - (geometry.size.height * self.splitRatio))
                
                Draw()
                    .environmentObject(self.drawData)
                    .frame(width: geometry.size.height * self.splitRatio, height: geometry.size.height * self.splitRatio)
                    .border(Color.blue, width: 1)
            }
        }
    }
}


struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

