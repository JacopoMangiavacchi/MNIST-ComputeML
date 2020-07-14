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
    
    let splitRatio: CGFloat = 0.5 //0.2445
    
    var body: some View {
        GeometryReader { geometry in
            NavigationView {
                VStack(spacing: 0) {
                    Form {
                        Section(header: Text("Dataset")) {
                            HStack {
                                ProgressView("Training: \(self.mnist.trainingBatchCount)", value: Float(self.mnist.trainingBatchCount), total: Float(self.mnist.expextedTrainingSamples))
                                Spacer(minLength: 10.0)
                                ProgressView("Test: \(self.mnist.testBatchCount)", value: Float(self.mnist.testBatchCount), total: Float(self.mnist.expectedTestingSamples))
                                Spacer(minLength: 10.0)
                                Button(action: {
                                    self.mnist.asyncPrepareData()
                                }) {
                                    Text("Prepare")
                                }.disabled(self.mnist.dataPreparing || self.mnist.modelTraining)
                            }
                        }
                        Section(header: Text("Training")) {
                            Stepper(value: self.$mnist.epochs, in: 1...20, label: { Text("Epoch:  \(self.mnist.epochs)")})
                            HStack {
                                Text(self.mnist.trainingFeedback)
                                Spacer()
                                Button(action: {
                                    self.mnist.asyncTrainGraph()
                                }) {
                                    Text("Train")
                                }.disabled(self.mnist.dataPreparing || self.mnist.modelTraining || !self.isDataReady(trainingCount: self.mnist.trainingBatchCount, predictionCount: self.mnist.testBatchCount))
                            }
                        }
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
                                        self.prediction = "\(self.mnist.predict(data: data))"
                                    }
                            }
                        }
                    }.frame(width: geometry.size.width, height: geometry.size.height - (geometry.size.height * self.splitRatio))
                    
                    Draw()
                        .environmentObject(self.drawData)
                        .frame(width: geometry.size.height * self.splitRatio, height: geometry.size.height * self.splitRatio)
                        .border(Color.blue, width: 1)
                }
                .navigationBarTitle("MNIST MLCompute", displayMode: .inline)
            }
        }
    }
}


struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

