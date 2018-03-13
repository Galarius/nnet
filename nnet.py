# -*- coding: utf-8 -*-
#!/usr/bin/env python

import os
import numpy as np
from pylab import plt

def plot_map(xs, ys):
    plt.plot(xs, ys)
    plt.grid(True)
    plt.show()

# T0 = np.array((
# [0, 6], [1, 5], [2, 4], [3, 3], [4,2], [5,1], [6,0], 

# ), dtype=float) 

# input
X = np.array((
# For I trajectory
[1, 5], [2, 5], [6, 1], [4, 4], [5,3], [10,4], [13,4], 
[14,2], [10,5], [8, 5], [5, 2], [9,2], [1, 3], [2, 3], 
[3, 3], [11,4], [11,2], [14,4], [7,3], [7, 1], 
# For II trajectory
[1, 7], [1, 6], [2, 6], [2,8], [3,7], [4, 6], [4,10], 
[5, 7], [6, 5], [7, 7], [8,6], [9,9], [10,7], [10,9], 
[12,6], [13,5], [13,7], [14,5], [7,5], [11,5]
), dtype=float)
xPredicted = np.array(([4, 8]), dtype=float)
# output
y = np.array((
    [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],
    [1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]), dtype=float)

X = X / np.amax(X, axis=0)
xPredicted = xPredicted / np.amax(xPredicted, axis=0)
y = y / 100

np.random.seed(1)

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