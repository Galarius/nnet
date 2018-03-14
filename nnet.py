# -*- coding: utf-8 -*-
#!/usr/bin/env python

import os
import numpy as np
from map import Map

map = Map(15, 10)
(d0, d1) = map.build(20)

# input
X = np.concatenate((d0, d1), axis=0)

# output
yd0 = np.array([[0]] * d0.shape[0], dtype=float)
yd1 = np.array([[1]] * d1.shape[0], dtype=float)
y = np.concatenate((yd0, yd1), axis=0)

# normalization
X = X / np.amax(X, axis=0)
# y = y / 100

# input for prediction
xPredicted = np.array(([4, 8]), dtype=float)
xPredicted = xPredicted / np.amax(xPredicted, axis=0)

class Neural_Network(object):
    def __init__(self):
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)
    
    def forward(self, X):
        self.z = np.dot(X, self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        o = self.sigmoid(self.z3)
        return o

    def backward(self, X, y, o):
        self.o_error = y - o
        self.o_delta = self.o_error * self.sigmoidPrime(o)

        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)

        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.o_delta)

    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)
    
    def predict(self, X):
        return self.forward(X)

    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    def sigmoidPrime(self, s):
        # derivative of sigmoid
        return s * (1 - s)

    def saveWeights(self):
        np.savetxt("w1.txt", self.W1, fmt="%s")
        np.savetxt("w2.txt", self.W2, fmt="%s")
    
    def loadWeights(self):
        self.W1 = np.loadtxt("w1.txt", dtype=float)
        self.W2 = np.loadtxt("w2.txt", dtype=float)

NN = Neural_Network()
if not os.path.exists("w1.txt") or \
   not os.path.exists("w2.txt"):
    for i in xrange(1000):
        print "Input: \n" + str(X)
        print "Actual Output: \n" + str(y)
        print "Predicted Output: \n" + str(NN.forward(X))
        print "Loss: \n" + str(np.mean(np.square(y - NN.forward(X))))
        print "\n"
        NN.train(X, y)
    NN.saveWeights()
else:
    NN.loadWeights()

print "Predicted data based on trained weights: "
print "Input (scaled): \n" + str(X)
print "Output: \n" + str(NN.predict(xPredicted))