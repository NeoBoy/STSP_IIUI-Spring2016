# -*- coding: utf-8 -*-
"""
The goal of this file is to train a Logistic Regression Classifier for the 
MNIST Dataset

This code has been written to work in Python 3

@author: Sharjeel Abid Butt
"""


#import sys, os, re, math
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
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")



def dataExtraction(data = 'train', class1 = 1, class0 = 0):
    import pickle, gzip
    # Load the dataset
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()
    
    if data is 'test':
        [data, labels] = test_set
    else:
        data   = np.concatenate((train_set[0], valid_set[0]), axis = 0)
        labels = np.concatenate((train_set[1], valid_set[1]), axis = 0)
        
    y1 = np.extract(labels == class1, labels)
    X1 = data[labels == class1, :]
    
    y0 = np.extract(labels == class0, labels)
    X0 = data[labels == class0, :]

    y = np.concatenate((y1, y0), axis = 0)
    X = np.concatenate((X1, X0), axis = 0)
    
    #X = (X - np.mean(X, axis = 0)) / (1 + np.std(X, axis = 0)) # Data Normalization
    y[y == class1] = 1
    y[y == class0] = 0
    y = np.reshape(y, (np.shape(X)[0], 1))
    return y, X

def sigmoid(z):
    S = 1.0 / (1.0 + np.exp(-z))
    return S

def logisticRegression(theta, X, y, Lambda):
    z = np.dot(X, theta)
    y_hat = sigmoid(z)
    f = - np.sum(y * np.log(y_hat) + (1.0 - y) * np.log(1.0 - y_hat)) / np.shape(X)[0]
    g = np.dot(X.T, y_hat - y)
    
    if Lambda != 0:
        f     += Lambda / 2 * np.sum(theta[1:] ** 2) / np.shape(X)[0]
        g[1:] += Lambda * theta[1:]
    
    return f, g

tic()

class1 = 1;
class0 = 0;

y, X     = dataExtraction('train', class1, class0)
theta    = np.random.rand(np.shape(X)[1] + 1, 1) * 1e-2
Xconcate = np.concatenate((np.ones((np.shape(X)[0], 1)), X), axis = 1)

noOfIter     = 5000
learningRate = 1e-1
stopVal      = 1e-3
Lambda       = 0.1

for i in range(0, noOfIter):
    [f, g] = logisticRegression(theta, Xconcate, y, Lambda)
    
    if not np.isnan(f):
        print(f, i)
    if f < stopVal:
        break
    else:
        theta -= learningRate * (1.0 / np.shape(X)[0]) * g


y_hatTrain = sigmoid(np.dot(Xconcate, theta)) > 0.5

error_train = np.sum(y != y_hatTrain) * 100.0 / np.size(y)


yTest, XTest = dataExtraction('test', class1, class0)
XconcateTest = np.concatenate((np.ones((np.shape(XTest)[0], 1)), XTest), axis = 1)
y_hatTest = sigmoid(np.dot(XconcateTest, theta)) > 0.5

error_test = np.sum(yTest != y_hatTest) * 100.0 / np.size(yTest)

toc()

print('\n\n\n')
print("Training Accuracy = " + str(100 - error_train) + "%")
print("Test Accuracy     = " + str(100 - error_test) + "%")
