import pandas as pd
import random
import numpy as np
import csv
import sys
import textwrap

class PerceptronModel():

    # Activation Function
    def stepFunction(self, t):
        if t >= 0.0:
            return 1
        return 0

    # Predict Using weights and Step Function
    def predict(self, X, Wts): 
        result = 0
        for i in range(0, len(X)):
            result = result + float(X[i])*Wts[i]
        return self.stepFunction(result)

    # Read Stored Weights from file for testing
    def getWeights(self):
        weights = pd.read_csv('weights.csv')
        return list(map(float, weights))

    # Writing Weights in file
    def storeWeights(self, wts):
        with open('weights.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(wts)

    # Updating Weights according to fromula
    def updateWeights(self, err, input, W, alpha):
        for i in range(0, len(W)):
            W[i] = W[i] + alpha*err*float(input[i])

    # Testing model 
    def testModel(self, inputs, desiredOutput):
        W = self.getWeights()
        tp = 0
        fp = 0
        fn = 0 
        tn = 0
        for i in range(0, len(desiredOutput)):
            predicted = self.predict(inputs.iloc[i], W)
            if predicted == desiredOutput[i]:
                if predicted == 1:
                    tp = tp + 1
                else:
                    tn = tn + 1
            else:
                if predicted == 0:
                    fn = fn + 1
                else:
                    fp = fp + 1

        self.writingModelTestResults(str(self.accuracy((tp + tn), len(desiredOutput))), str(self.precision(tp, fp)), str(self.recall(tp, fn)),tp,tn,fp,fn)

    # Finding Recall of model
    def recall(self, tp, fn):
        return tp / (tp + fn)

    # Finding Precision of Model
    def precision(self, tp, fp):
        return tp / (tp + fp)

    # Finding Accuracy of Model
    def accuracy(self, correct, total):
        return (correct/total) * 100

    # Traing Model from initial Weights and update them accordingly
    def training(self, inputs, desiredOutput, W, alpha = 0.1): 
        self.initializeFile()

        epoch = 1
        commulativeError = 1

        while commulativeError != 0 and epoch != 1000:
            commulativeError = 0

            for i in range(0, len(desiredOutput)):
                # Predicting on initial Weights
                actualOutput = self.predict(inputs.iloc[i], W)
                # Finding Error
                error = float(desiredOutput[i]) - actualOutput
                initialW = W
                # Updating Weights if error is present
                if error != 0:
                    commulativeError += 1
                    self.updateWeights(error, inputs.iloc[i], W, alpha)
                # Writing in to File
                self.writingResults(epoch, inputs.iloc[i], desiredOutput[i], initialW, actualOutput, error, W)

            epoch += 1
        # Finishing trainig and storing weighs in local storage
        self.storeWeights(W)

    # Writing results of model in file
    def writingModelTestResults(self, accuracy, precision, recall, tp, tn, fp, fn):
        f = open('modelTestResults.txt','w')

        f.write('Confusion Matrix\n')
        f.write('True Positive\tTrue Negative\tFalse Positive\tFalse Negative\n')
        f.write('\t\t' + str(tp) + '\t\t\t' + str(tn) + '\t\t\t\t' + str(fp) + '\t\t\t\t' + str(fn) + '\n') 
        f.write('The Model is ' + str(accuracy) + '% Accurate\n')
        f.write('The Model Precision is ' + str(precision) + '\n')
        f.write('The Model Recall is ' + str(recall) + '\n')

        f.close()

    # Writing boiler plate
    def initializeFile(self):
        f = open('modelLearning.txt','w')

        f.write('Epoch  \t\t\t\t  Inputs \t\t\t\t Output \t\t\t\t Weights \t\t\t\t\t Prediction  \t\t  Error \t\t\t\t Final Weights\n')
       
        f.close()   

    # Writing Traing process in file.
    def writingResults(self, epoch, inp, des, RW, pred, err, FW):
        f = open('modelLearning.txt', 'a')

        f.write(str(epoch) + '                ')
        for i in range(0, len(RW)):
            f.write(str(inp[i]) + ',')
        f.write('           ' + str(des) + '         ')
        for i in range(0, len(RW)):
            f.write(str(round(RW[i], 3)) + ',')
        f.write('             ' + str(pred) + '     ')
        f.write('         ' + str(err) + '          ')
        for i in range(0 , len(FW)):
            f.write(str(round(FW[i], 3)) + ',')
        f.write('\n')

        f.close()


# Data Preprocessing

model = PerceptronModel()

def preprocess(file, proc): 
    # Read Dataset
    flowerData = pd.read_csv(file, usecols=['x1', 'x2', 'x3', 'x4', 'output'])
    # Create the input matrix
    inputs = flowerData.drop(['output'], axis = 1)
    inputs['bx'] = 1
    # Create the output array
    desiredOutput = flowerData['output']
    desiredOutput = desiredOutput.replace({'Iris-setosa' : 0})
    desiredOutput =  desiredOutput.replace({'Iris-versicolor' : 1})
    # Initialize random Weights
    W = [ random.uniform(-0.5, 0.5) for i in range(0, len(flowerData.columns)) ]  
    # Checking Which Function to run
    if proc == 'train':
        model.training(inputs, desiredOutput, W)
    else:
        model.testModel(inputs, desiredOutput)    



preprocess(sys.argv[1], sys.argv[2])




# And Gate data processing

# gateData = pd.read_csv('AndGate.csv', usecols=['X', 'Y', 'Output'])
# inputs = gateData.drop(['Output'], axis = 1)
# inputs['bx'] = 1
# desiredOutput = gateData['Output']
# W = [ random.uniform(-0.5, 0.5) for i in range(0, len(gateData.columns)) ]


