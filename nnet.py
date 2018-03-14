# -*- coding: utf-8 -*-
#!/usr/bin/env python

import os
import numpy as np
from map import Map

map = Map(15, 10)
(d0, d1) = map.build(100)

# input
X = np.concatenate((d0, d1), axis=0)

# output
yd0 = np.array([[0]] * d0.shape[0], dtype=float)
yd1 = np.array([[1]] * d1.shape[0], dtype=float)
y = np.concatenate((yd0, yd1), axis=0)

# normalization
X = X / np.amax(X, axis=0)
# y - normalized

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

np.set_printoptions(precision=2, suppress=True)

NN = Neural_Network()
if not os.path.exists("w1.txt") or \
   not os.path.exists("w2.txt"):
    for i in xrange(1000):
        print "Input: \n" + str(X)
        print "Actual Output: \n" + str(y)
        print "Predicted Output: \n" 
        print np.round(NN.forward(X))
        print "Loss: \n" + str(np.mean(np.square(y - NN.forward(X))))
        print "\n"
        NN.train(X, y)
    NN.saveWeights()
else:
    NN.loadWeights()

# input for prediction
zd0, zd1 = map.dataset(0, 2), map.dataset(1, 8)
z = np.concatenate((zd0, zd1), axis=0)
z_scaled = z / np.amax(z, axis=0)
# check predicted
z_check0 = np.array([0] * zd0.shape[0], dtype=float)
z_check1 = np.array([1] * zd1.shape[0], dtype=float)
z_check = np.concatenate((z_check0, z_check1), axis=0)

print "Predicted data based on trained weights: "
print "Input (scaled): \n" + str(X)
print "Input for prediction: \n" + str(z)
print "Actual:"
print z_check
print "Output:"
res = np.round(NN.predict(z_scaled))
print res

if (res == z_check).all():
    print "All good!"
else:
    print "Correct:"
    print res == z_check

map.plot(z, True)