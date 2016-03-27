# -*- coding: utf-8 -*-
"""
The goal of this file is to design a class for Neural Networks

@author: Sharjeel
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
        
#        self.Nstruct   = [noOfInputs, nodesInEachLayer, noOfOutputs]
        self.Theta     = []
        lmin, lmax     = parametersRange
        
        for ind in range(noOfLayers + 1):
            if ind == 0:
                tempTheta = self.randTheta(lmin, lmax, noOfInputs, self.n_H[ind])
            elif ind == noOfLayers:
                tempTheta = self.randTheta(lmin, lmax, self.n_H[-1], noOfOutputs)
            else:
                tempTheta = self.randTheta(lmin, lmax, self.n_H[ind - 1], self.n_H[ind])
            
            self.Theta.append(tempTheta)
    
    def __str__(self):
        return "This neural network has a " + str(self.n_I) + ' X ' + \
        str(self.n_H) + ' X ' + str(self.n_O) + " structure."
        
    def randTheta(self, l_min, l_max, i_nodes, o_nodes):
        theta = l_min + np.random.rand(i_nodes + 1, o_nodes) * (l_max - l_min)
        return theta
        
    
    def forward_pass(self):
        raise NotImplementedError
    
    def backward_pass(self):
        raise NotImplementedError
    
    def trainNNET(self, in_data, out_data, stoppingCriteria = 1e-3, 
                  Lambda = 0, noOfIterations = 1000):
        raise NotImplementedError
    