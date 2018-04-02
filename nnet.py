# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np

class NeuralNetwork(object):
    def __init__(self, n_in, n_hl, n_out, use_bias = False):
        self.n_in  = n_in
        self.n_hl  = n_hl
        self.n_out = n_out

        self.w1 = np.random.randn(self.n_in, self.n_hl)
        self.w2 = np.random.randn(self.n_hl, self.n_out)
        self.bias = use_bias if np.random.uniform() else 0
    
    def forward(self, X):
        """
        Прямое распространение
        """
        self.z = np.dot(X, self.w1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.w2) + self.bias
        o = self.sigmoid(self.z3)
        return o

    def backward(self, X, t, y):
        """
        Обратное распространение
        """
        self.o_error = t - y
        self.o_delta = self.o_error * self.sigmoidPrime(y)

        self.z2_error = self.o_delta.dot(self.w2.T)
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)

        self.w1 += X.T.dot(self.z2_delta)   # gradient
        self.w2 += self.z2.T.dot(self.o_delta)

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
        np.savetxt("w1.txt", self.w1, fmt="%s")
        np.savetxt("w2.txt", self.w2, fmt="%s")
    
    def loadWeights(self):
        self.w1 = np.loadtxt("w1.txt", dtype=float)
        self.w2 = np.loadtxt("w2.txt", dtype=float)
        self.w2 = self.w2.reshape(self.w2.shape[0], 1)
        
