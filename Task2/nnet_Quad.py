# -*- coding: utf-8 -*-
"""
The goal of this file is to design a class for Neural Networks

@author: Sharjeel Abid Butt

@References

1. http://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
2. https://github.com/NeoBoy/STSP_IIUI-Spring2016/tree/master/Task2

"""

import copy

import numpy as np
#import scipy as sp
#import scipy.stats as stats
#import pandas as pd
#import matplotlib as mpl
import matplotlib.pyplot as plt

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
        
    def setTheta(self, thetaLayer = 1, thetaIndex = [1, 0], allSet = False):
        """
        Updates the Theta values of the Neural Network
        """
        if allSet:
            for l in range(self.n_L + 1):
                print '\n\nEnter Theta values for Layer ' + str(l + 1) + ':\n'
                in_nodes, out_nodes = np.shape(self.Theta[l])
                for inIndex in range(in_nodes):
                    for outIndex in range(out_nodes):
                        self.Theta[l][inIndex][outIndex] = float (raw_input( \
                        'Enter Theta[' + str(outIndex + 1) + '][' + str(inIndex) +']:'))
        else:
            outIndex, inIndex = thetaIndex
            self.Theta[thetaLayer - 1][inIndex][outIndex - 1] = float (raw_input( \
                        'Enter Theta[' + str(outIndex) + '][' + str(inIndex) +']:'))
        
        print '\n\n\nTheta Update Complete.\n\n\n'
        
    def getTheta(self):
        return copy.deepcopy(self.Theta)
    
    def forward_pass(self, nodes, X, y):
        """
        Does the forward pass stage of Backpropagation
        """
#        raise NotImplementedError
        m = np.shape(y)[0]
        
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
        """        
        Does the Backpass stage of Backpropagation        
        """
#        raise NotImplementedError    
        m = np.shape(y)[0]
        
        if self.a_Func == 'sigmoid':
            delta[-1] = (nodes[-1] - y) * self.sigmoid(nodes[-1], True)
            for l in range(self.n_L - 1, -1, -1):
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
            
        return grad, delta
                
    
    def trainNNET(self, data, labels, stoppingCriteria = 1e-3, LearningRate = 1e-1, 
                  Lambda = 0, noOfIterations = 1000, moreDetail = False):
        """
        Does the training of the Neural Network
        """
#        raise NotImplementedError
        if (np.shape(data)[0]   != np.shape(labels)[0] or \
            np.shape(data)[1]   != self.n_I or \
            np.shape(labels)[1] != self.n_O):
                raise ValueError('Data is not suitable for this neural network')
        
        m     = np.shape(labels)[0]
        nodes = []
        delta = []
        grad  = []
        eV    = []
        
        print 'Training Started:'
        for l in range(self.n_L + 1):            
            nodes.append(np.zeros((m, self.nodes[l])))
            delta.append(np.zeros((m, self.nodes[l])))
            grad.append(np.shape(self.Theta[l]))

        print "Epoch \t Error"
        for epoch in range(noOfIterations):
            nodes = self.forward_pass(nodes, data, labels)
            
            labels_hat = nodes[-1]
            error = np.sum((labels_hat - labels) ** 2) / (2.0 * m)           
            if Lambda != 0:
                for l in range(self.n_L + 1):
                    error += Lambda / 2 * np.sum(self.Theta[l][1:] ** 2) / m
                
            print str(epoch) + " \t " + str(np.nan_to_num(error))
            
            eV.append(error)
            
            if error <= stoppingCriteria:
                break
            else:
                grad, delta = self.backward_pass(delta, nodes, data, \
                                                 labels, grad, Lambda)
                
                for l in range(self.n_L + 1):
                    self.Theta[l] -= LearningRate / m * grad[l]
        
        if moreDetail:
            return eV, nodes, grad, delta
            
        return eV
        

    def predictNNET(self, data, labels):
        nodes = []        
        m     = np.shape(labels)[0]
        for l in range(self.n_L + 1):            
            nodes.append(np.zeros((m, self.nodes[l])))
        
        nodes = self.forward_pass(nodes, data, labels)
        
        labels_hat = nodes[-1] > 0.5
        return labels_hat    
     




# Main Code starts here
inData  = np.array([[0.05, 0.10]])
outData = np.array([[0.01, 0.99]])

Q2 = nnet(noOfInputs = 2, noOfLayers = 1, nodesInEachLayer = [2], noOfOutputs = 2)

Q2.Theta = [np.array([[0.35, 0.35], [0.15, 0.20], [0.25, 0.30]]), \
            np.array([[0.60, 0.60], [0.40, 0.45], [0.50, 0.55]])]

#Q2.setTheta(allSet = True)
original_theta = Q2.getTheta()

loss, nodes, grad, delta = Q2.trainNNET(inData, outData, LearningRate = 0.5, \
                                        noOfIterations = 1, moreDetail = True)

updated_Theta = Q2.getTheta()

