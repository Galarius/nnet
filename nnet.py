# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np

class NeuralNetwork(object):
    def __init__(self, n_in, n_hl, n_out, use_bias = False):
        self.n_in  = n_in
        self.n_hl  = n_hl
        self.n_out = n_out

        self.w_in_h = np.random.randn(self.n_in, self.n_hl)
        self.w_h_out = np.random.randn(self.n_hl, self.n_out)
        self.bias = use_bias if np.random.uniform() else 0
    
    def forward(self, x):
        """
        Прямое распространение
        """
        self.z = np.dot(x, self.w_in_h)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.w_h_out) + self.bias
        o = self.sigmoid(self.z3)
        return o

    def backward(self, x, t, y):
        """
        Обратное распространение
        """
        self.o_error = t - y
        self.o_delta = self.o_error * self.sigmoidPrime(y)

        self.z2_error = self.o_delta.dot(self.w_h_out.T)
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)

        self.w_in_h += x.T.dot(self.z2_delta)   # gradient
        self.w_h_out += self.z2.T.dot(self.o_delta)

    def train(self, x, t):
        o = self.forward(x)
        self.backward(x, t, o)
    
    def predict(self, x):
        return self.forward(x)

    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    def sigmoidPrime(self, s):
        """
        Derivative of sigmoid
        """
        return s * (1 - s)

    def saveWeights(self):
        np.savetxt("w_in_h.txt", self.w_in_h, fmt="%s")
        np.savetxt("w_h_out.txt", self.w_h_out, fmt="%s")
    
    def loadWeights(self):
        self.w_in_h = np.loadtxt("w_in_h.txt", dtype=float)
        self.w_h_out = np.loadtxt("w_h_out.txt", dtype=float)
        self.w_h_out = self.w_h_out.reshape(self.w_h_out.shape[0], 1)
        
