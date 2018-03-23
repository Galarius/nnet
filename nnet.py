# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np

class NeuralNetwork(object):
    def __init__(self, ni, no):
        self.inputSize  = ni
        self.outputSize = no
        self.hiddenSize = ni + 1
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)
    
    def forward(self, X):
        """
        Прямое распространение
        """
        self.z = np.dot(X, self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        o = self.sigmoid(self.z3)
        return o

    def backward(self, X, t, y):
        """
        Обратное распространение
        """
        self.o_error = t - y
        self.o_delta = self.o_error * self.sigmoidPrime(y)

        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)

        self.W1 += X.T.dot(self.z2_delta)   # gradient
        self.W2 += self.z2.T.dot(self.o_delta)

    def train(self, X, t):
        o = self.forward(X)
        self.backward(X, t, o)
    
    def predict(self, X):
        return self.forward(X)

    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    def sigmoidPrime(self, s):
        """
        Derivative of sigmoid
        """
        return s * (1 - s)

    def saveWeights(self):
        np.savetxt("w1.txt", self.W1, fmt="%s")
        np.savetxt("w2.txt", self.W2, fmt="%s")
    
    def loadWeights(self):
        self.W1 = np.loadtxt("w1.txt", dtype=float)
        self.W2 = np.loadtxt("w2.txt", dtype=float)
        self.W2 = self.W2.reshape(self.W2.shape[0], 1)
        
