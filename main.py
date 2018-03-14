# -*- coding: utf-8 -*-
#!/usr/bin/env python

__author__ = "Ilya Shoshin"
__copyright__   = "MAI, M30-102-17"

import os
import numpy as np

from nnet import NeuralNetwork
from map import Map

def main():
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

    np.set_printoptions(precision=2, suppress=True)

    NN = NeuralNetwork()
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

if __name__ == "__main__":
    main()