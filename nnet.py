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
        "Прямое распространение"
        # скалярное произведение входных данных и весов
        self.s_h = np.dot(x, self.w_in_h) 
        # активация нейронов скрытого слоя
        self.y_h = self.sigmoid(self.s_h) 
        # скалярное произведение активированных нейронов
        # скрытого слоя и весов + вес синопса от нейрона смещения
        self.s_o = np.dot(self.y_h, self.w_h_out) + self.bias
        # активация нейронов выходного слоя
        return self.sigmoid(self.s_o)

    def backward(self, x, t, y):
        "Обратное распространение"
        self.o_error = t - y
        self.o_delta = self.o_error * self.sigmoidPrime(y)

        self.z2_error = self.o_delta.dot(self.w_h_out.T)
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.y_h)

        self.w_in_h += x.T.dot(self.z2_delta)   # gradient
        self.w_h_out += self.y_h.T.dot(self.o_delta)

    def train(self, x, t):
        o = self.forward(x)
        self.backward(x, t, o)
    
    def predict(self, x):
        return self.forward(x)

    def sigmoid(self, s):
        "Сигмоидальная логистическая функция активации"
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
        
