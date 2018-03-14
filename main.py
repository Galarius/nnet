# -*- coding: utf-8 -*-
#!/usr/bin/env python

__author__ = "Ilya Shoshin"
__copyright__   = "MAI, M30-102-17"

import os
import sys
import numpy as np

from nnet import NeuralNetwork
from map import Map

np.set_printoptions(precision=2, suppress=True)

def train_data(map):
    (d0, d1) = map.build(40)
    # input
    X = np.concatenate((d0, d1), axis=0)
    # targets
    yd0 = np.array([[0]] * d0.shape[0], dtype=float)
    yd1 = np.array([[1]] * d1.shape[0], dtype=float)
    y = np.concatenate((yd0, yd1), axis=0)
    # normalization
    X = X / np.amax(X, axis=0)
    # y already normalized
    return (X, y)

def help(name):
    print "{} [-t,-h]".format(name)

def main(argv):
    # if not len(argv):
    #     help(sys.argv[0])
    #     sys.exit(1)
    
    opt_train = False

    for arg in argv:
        if arg == '-h':
            help(sys.argv[0])
            sys.exit(0)
        elif arg == '-t':
            opt_train = True
        else:
            print 'Unrecognized option {}'.format(arg)
            help(sys.argv[0])
            sys.exit(2)

    map = Map(15, 10)
    X, y = train_data(map)

    NN = NeuralNetwork()
    if  opt_train or \
        not os.path.exists("w1.txt") or \
        not os.path.exists("w2.txt"):
        print 'Training...'
        for i in xrange(1000):
            #print "Input: \n" + str(X)
            #print "Actual Output: \n" + str(y)
            #print "Predicted Output: \n" 
            #print np.round(NN.forward(X))
            #print "Loss: \n" + str(np.mean(np.square(y - NN.forward(X))))
            #print "\n"
            NN.train(X, y)
        NN.saveWeights()
        print 'Done.'
    else:
        NN.loadWeights()

    # input for prediction
    zd0, zd1 = map.dataset(0, 2), map.dataset(1, 8)
    z = np.concatenate((zd0, zd1), axis=0)
    z_scaled = z / np.amax(z, axis=0)
    # check predicted
    z_check0 = np.array([[0]] * zd0.shape[0], dtype=float)
    z_check1 = np.array([[1]] * zd1.shape[0], dtype=float)
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
        print "\nAll Good!"
    else:
        print "Correct:"
        print res == z_check

    map.plot(z, True)

if __name__ == "__main__":
    main(sys.argv[1:])