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
    print "{} [-h,-s,-t,-l]".format(name)

def main(argv):    
    opt_train = False
    opt_log = False
    opt_seed = False
    N = 500

    for arg in argv:
        if arg == '-h':
            help(sys.argv[0])
            sys.exit(0)
        elif arg == '-s':
            opt_seed = True
        elif arg == '-t':
            opt_train = True
        elif arg == '-l':
            opt_log = True
        else:
            print 'Unrecognized option {}'.format(arg)
            help(sys.argv[0])
            sys.exit(2)

    if opt_seed:
        np.random.seed(1)
    map = Map(20, 20)
    X, y = train_data(map)

    NN = NeuralNetwork(2, 1)
    if  opt_train or \
        not os.path.exists("w1.txt") or \
        not os.path.exists("w2.txt"):
        print 'Training...'
        if opt_log:
            with open('training.log', 'w') as f:
                for epoch in xrange(N):
                    f.write('Epoch {}\n'.format(epoch))
                    f.write("Input:\n{}\n".format(X.T))
                    f.write("Actual Output:\n{}\n".format(y.T))
                    f.write("Predicted Output:\n{}\n".format(np.round(NN.forward(X).T)))
                    f.write("Loss:\n{}\n\n".format(str(np.mean(np.square(y - NN.forward(X))))))
                    NN.train(X, y)
        else:
            for epoch in xrange(N):
                NN.train(X, y)
        NN.saveWeights()
        print 'Done.'
    else:
        NN.loadWeights()

    # input for prediction
    if opt_seed:
        np.random.seed(2)
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
        print "{}% are good!".format((res == z_check).sum() * 100 / len(res))

    # map.plotMap('plt_map.png')
    map.plot(z, 'map.png')

if __name__ == "__main__":
    main(sys.argv[1:])