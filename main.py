# -*- coding: utf-8 -*-
#!/usr/bin/env python

__author__ = "Ilya Shoshin"
__copyright__   = "MAI, M30-102-17"

import os
import argparse
import numpy as np

from nnet import NeuralNetwork
from map import Map

np.set_printoptions(precision=2, suppress=True)

def train_data(map):
    (d0, d1) = map.build(40)
    # input
    x = np.concatenate((d0, d1), axis=0)
    # targets
    td0 = np.array([[0]] * d0.shape[0], dtype=float)
    td1 = np.array([[1]] * d1.shape[0], dtype=float)
    t = np.concatenate((td0, td1), axis=0)
    # normalization
    x = x / np.amax(x, axis=0)
    # t already normalized
    return (x, t)

def prediction_data(map, max_n = 50):
    n = np.sqrt(max_n)
    (zd0, zd1) = map.dataset(0, np.random.randint(n)), map.dataset(1, np.random.randint(n))
    z = np.concatenate((zd0, zd1), axis=0)
    z_scaled = z / np.amax(z, axis=0)
    # check predicted
    z_check0 = np.array([[0]] * zd0.shape[0], dtype=float)
    z_check1 = np.array([[1]] * zd1.shape[0], dtype=float)
    z_check = np.concatenate((z_check0, z_check1), axis=0)
    return (x, t)

def main(argv):    
    if args.seed:
        np.random.seed(args.seed)

    map = Map(20, 20)
    x, t = train_data(map)

    NN = NeuralNetwork(2, 3, 1, True)
    if  args.train or \
        not os.path.exists("w1.txt") or \
        not os.path.exists("w2.txt"):
        print 'Training...'
        if args.logging:
            with open('training.log', 'w') as f:
                for epoch in xrange(args.epochs):
                    f.write('Epoch {}\n'.format(epoch))
                    f.write("Input:\n{}\n".format(x.T))
                    f.write("Actual Output:\n{}\n".format(t.T))
                    f.write("Predicted Output:\n{}\n".format(np.round(NN.forward(x).T)))
                    f.write("Loss:\n{}\n\n".format(str(np.mean(np.square(t - NN.forward(x))))))
                    NN.train(x, t)
        else:
            for epoch in xrange(args.epochs):
                NN.train(x, t)
        NN.saveWeights()
        print 'Done.'
    else:
        NN.loadWeights()

    # input for prediction
    if args.seed:
        np.random.seed(args.seed + 1)
    zd0, zd1 = map.dataset(0, 2), map.dataset(1, 8)
    z = np.concatenate((zd0, zd1), axis=0)
    z_scaled = z / np.amax(z, axis=0)
    # check predicted
    z_check0 = np.array([[0]] * zd0.shape[0], dtype=float)
    z_check1 = np.array([[1]] * zd1.shape[0], dtype=float)
    z_check = np.concatenate((z_check0, z_check1), axis=0)

    print "Predicted data based on trained weights: "
    print "Input (scaled): \n" + str(x)
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

    if args.plotting:
        # map.plotMap('plt_map.png')
        map.plot(z, 'map.png')

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-t', '--train', action='store_true', help='perform training')
    ap.add_argument('-e', '--epochs', type=int, default=1000, help='train with specified number of epochs')
    ap.add_argument('-a', '--alpha', type=float, default=0.01, help='gradient descent momentum')
    ap.add_argument('-s', '--seed',  type=int, default=0, help='seed random generator')
    ap.add_argument('-l', '--logging', action='store_true', help='write training process into training.log file')
    ap.add_argument('-p', '--plotting', action='store_true', help='show plot')
    args = ap.parse_args()
    main(args)