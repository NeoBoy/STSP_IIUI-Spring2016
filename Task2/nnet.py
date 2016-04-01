# -*- coding: utf-8 -*-
"""
The goal of this file is to design a class for Neural Networks

@author: Sharjeel Abid Butt

@References

1. http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm
2. https://grantbeyleveld.wordpress.com/2015/10/09/implementing-a-artificial-neural-network-in-python/
3. http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch
"""

import mnist_load as mload

import numpy as np
#import scipy as sp
#import scipy.stats as stats
#import pandas as pd
#import matplotlib as mpl
#import matplotlib.pyplot as plt

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
    else:
        print "Toc: start time not set"

def dataExtraction(data = 'train', class1 = 1, class0 = 0):
    
    [data, labels] = mload.load(data)
    y1 = np.extract(labels == class1, labels)
    X1 = data[labels == class1, :, :]
    
    y0 = np.extract(labels == class0, labels)
    X0 = data[labels == class0, :, :]

    y = np.concatenate((y1, y0), axis = 0)
    X = np.concatenate((X1, X0), axis = 0)
    
    X = np.reshape(X, (np.shape(X)[0],np.shape(X)[1] * np.shape(X)[2]))
    X = (X - np.mean(X, axis = 0)) / (1 + np.std(X, axis = 0)) # Data Normalization
    y[y == class1] = 1
    y[y == class0] = 0
    y = np.reshape(y, (np.shape(X)[0], 1))
    return y, X



class nnet(object):
    """
    A class that implements Basic Neural Networks Architecture
    """
    
    def __init__(self, noOfInputs = 2, noOfLayers = 2, nodesInEachLayer = [2, 2], 
               noOfOutputs = 2, activationFunction = 'sigmoid', 
               parametersRange = [-1, 1]):
        """
        Creates a Neural Network
        """
        if (len(nodesInEachLayer) != noOfLayers):
            raise ValueError('Incorrect Parameters provided!')
        
        self.n_I       = noOfInputs
        self.n_L       = noOfLayers
        self.n_H       = nodesInEachLayer
        self.n_O       = noOfOutputs
        self.a_Func    = activationFunction
        self.pR        = parametersRange
        
        #self.Nstruct   = [noOfInputs, nodesInEachLayer, noOfOutputs]
        self.Theta     = []
        self.nodes     = []
#        self.nodes.append(np.zeros((noOfInputs, 1)))
        lmin, lmax     = parametersRange
        
        
        for l in range(noOfLayers + 1):
            if l == 0:
                tempTheta = self.randTheta(lmin, lmax, noOfInputs, self.n_H[l])
            elif l == noOfLayers:
                tempTheta = self.randTheta(lmin, lmax, self.n_H[-1], noOfOutputs)
            else:
                tempTheta = self.randTheta(lmin, lmax, self.n_H[l - 1], self.n_H[l])
            
            tempNode  = np.shape(tempTheta)[1]
            
            self.Theta.append(tempTheta)
            self.nodes.append(tempNode)
    
    def __str__(self):
        return "This neural network has a " + str(self.n_I) + ' X ' + \
        str(self.n_H) + ' X ' + str(self.n_O) + " structure."
        
    def randTheta(self, l_min, l_max, i_nodes, o_nodes):
        theta = l_min + np.random.rand(i_nodes + 1, o_nodes) * (l_max - l_min)
        return theta
    
    def sigmoid(self, z, derivative = False):
        if derivative:
            return z * (1 - z)
        S = 1.0 / (1.0 + np.exp(-z))
        return S    
    
    def forward_pass(self, nodes, X, y):
#        raise NotImplementedError
        m = np.size(y)
        
        for l in range(self.n_L + 1):
            if l == 0:
                node_in  = np.concatenate((np.ones((m, 1)), X), axis = 1)
            else:
                node_in  = np.concatenate((np.ones((m, 1)), nodes[l - 1]), axis = 1)
            node_out = np.dot(node_in, self.Theta[l])
                
            if self.a_Func == 'sigmoid':
                nodes[l] = self.sigmoid(node_out)
            
        return nodes
                        
    
    def backward_pass(self, delta, nodes, X, y, grad, Lambda):
#        raise NotImplementedError    
        m = np.size(y)
        
        if self.a_Func == 'sigmoid':
            delta[-1] = (nodes[-1] - y) * self.sigmoid(nodes[-1], True)
            for l in range(self.n_L - 1, 0, -1):
                delta[l] = np.dot(delta[l + 1], self.Theta[l + 1][1:].T) \
                           * self.sigmoid(nodes[l], True)

        for l in range(self.n_L + 1):
            if l == 0:
                Xconcate = np.concatenate((np.ones((m, 1)), X), axis = 1)
                grad[l]  = np.dot(Xconcate.T, delta[l]) 
            else:
                nodeConcated = np.concatenate((np.ones((m, 1)), nodes[l - 1]), axis = 1)
                grad[l]      = np.dot(nodeConcated.T, delta[l]) 
            
            if Lambda != 0:
                    grad[l][1:] += Lambda * self.Theta[l][1:]
            
        return grad
                
    
    def trainNNET(self, data, labels, stoppingCriteria = 1e-3, 
                  LearningRate = 1e-1, Lambda = 0, noOfIterations = 1000):
#        raise NotImplementedError
        if (np.shape(data)[0]   != np.shape(labels)[0] or \
            np.shape(data)[1]   != self.n_I or \
            np.shape(labels)[1] != self.n_O):
                raise ValueError('Data is not suitable for this neural network')
        
        m     = np.shape(data)[0]
        nodes = []
        delta = []
        grad  = []
        
        for l in range(self.n_L + 1):            
            nodes.append(np.zeros((m, self.nodes[l])))
            delta.append(np.zeros((m, self.nodes[l])))
            grad.append(np.shape(self.Theta[l]))

        print "Epoch \t Error"
        for epoch in range(noOfIterations):
            nodes = self.forward_pass(nodes, data, labels)
            
            labels_hat = nodes[-1]
#            error = - np.sum(labels * np.log(labels_hat) + \
#                      (1.0 - labels) * np.log(1.0 - labels_hat)) / m
            error = np.sum((labels_hat - labels) ** 2) / m           
            if Lambda != 0:
                for l in range(self.n_L + 1):
                    error += Lambda / 2 * np.sum(self.Theta[l][1:] ** 2) / m
                
            print str(epoch) + " \t " + str(np.nan_to_num(error))
            
            
            if error <= stoppingCriteria:
                break
            else:
                grad = self.backward_pass(delta, nodes, data, labels, grad, Lambda)
                
                for l in range(self.n_L + 1):
                    self.Theta[l] -= learningRate / m * grad[l]
        return error
        

    def predictNNET(self, data, labels):
        nodes = []        
        m     = np.size(labels)
        for l in range(self.n_L + 1):            
            nodes.append(np.zeros((m, self.nodes[l])))
        
        nodes = self.forward_pass(nodes, data, labels)
        
        labels_hat = nodes[-1] > 0.5
        return labels_hat    
#       




# Main Code starts here
class1 = 1
class0 = 0

labelsTrain, dataTrain = dataExtraction('train', class1, class0)

noOfIter     = 5000
learningRate = 1e1
stopVal      = 1e-3
Lambda       = 0.1
pR           = [-1, 1]

mnistClassifier = nnet(noOfInputs = 784, noOfLayers = 2, nodesInEachLayer = [50, 50], \
                       noOfOutputs = 1, parametersRange = pR)

tic()
loss = mnistClassifier.trainNNET(dataTrain, labelsTrain, noOfIterations = 1000, \
                                 Lambda = Lambda)
toc()

print "\n\n\n"

labels_hatTrain = mnistClassifier.predictNNET(dataTrain, labelsTrain)
Train_Accuracy  = np.sum(labels_hatTrain == labelsTrain) * 100.0 / np.size(labelsTrain)
print "Training Accuracy = " + str(Train_Accuracy) + "%"

labelsTest, dataTest = dataExtraction('test', class1, class0)

labels_hatTest = mnistClassifier.predictNNET(dataTest, labelsTest)
Test_Accuracy  = np.sum(labels_hatTest == labelsTest) * 100.0 / np.size(labelsTest)
print "Test Accuracy = " + str(Test_Accuracy) + "%"