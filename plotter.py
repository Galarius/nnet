# -*- coding: utf-8 -*-
#!/usr/bin/env python

__author__ = "Ilya Shoshin"
__copyright__   = "MAI, M30-102-17"

import numpy as np      
from pylab import plt   

def sigmoid(s, a = 1):
    return 1 / (1 + np.exp(- a * s))

def sigmoidTanh(s, a = 1):
    return np.tanh(s / a)

def plotSigmoidExp(fname=None):
    fig, ax = plt.subplots()
    xs = np.linspace(-10.0, 10.0, num = 50, endpoint=True)
    ys = [sigmoid(x, 0.9) for x in xs]
    ax.plot(xs, ys, 'black')
    plt.title("y=sigmoid(s)")
    plt.grid(True)
    if fname:
        plt.savefig(fname)
    plt.show()

def plotSigmoidTanh(fname=None):
    fig, ax = plt.subplots()
    xs = np.linspace(-10.0, 10.0, num = 50, endpoint=True)
    ys = [sigmoidTanh(x, 0.9) for x in xs]
    ax.plot(xs, ys, 'black')
    plt.title("y=sigmoid(s)")
    plt.grid(True)
    if fname:
        plt.savefig(fname)
    plt.show()

def plotSigmoid(fname=None):
    fig, ax = plt.subplots()
    xs = np.linspace(-10.0, 10.0, num = 50, endpoint=True)
    ys = [sigmoid(0.5 * x) for x in xs]
    ax.plot(xs, ys, 'r', label='sig(0.5 * x)')
    ys = [sigmoid(1.0 * x) for x in xs]
    ax.plot(xs, ys, 'g--', label='sig(1.0 * x)')
    ys = [sigmoid(2.5 * x) for x in xs]
    ax.plot(xs, ys, 'b.', label='sig(2.5 * x)')
    legend = ax.legend(loc='best', framealpha=0.5)
    plt.title("y=sigmoid(Input * W1)")
    plt.grid(True)
    if fname:
        plt.savefig(fname)
    plt.show()

def plotSigmoidBias(fname=None):
    fig, ax = plt.subplots()
    xs = np.linspace(-10.0, 10.0, num = 50, endpoint=True)
    ys = [sigmoid(1.0 * x - 5.0) for x in xs]
    ax.plot(xs, ys, 'r', label='sig(1.0 * x - 5 * 1.0)')
    ys = [sigmoid(1.0 * x) for x in xs]
    ax.plot(xs, ys, 'g--', label='sig(1.0 * x + 0 * 1.0)')
    ys = [sigmoid(1.0 * x + 5.0) for x in xs]
    ax.plot(xs, ys, 'b.', label='sig(1.0 * x + 5 * 1.0)')
    legend = ax.legend(loc='best', framealpha=0.5)
    plt.title("y=sigmoid(Input * W1 + 1.0 * W2)")
    plt.grid(True)
    if fname:
        plt.savefig(fname)
    plt.show()

plotSigmoidExp('plt_sig_e.png')
plotSigmoidTanh('plt_sig_tanh.png')
plotSigmoid('plt_sig_w1.png')
plotSigmoidBias('plt_sig_w1w2.png')