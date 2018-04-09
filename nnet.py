# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np

class NeuralNetwork(object):
    def __init__(self, n_in, n_hl, n_out, n_hlayers, use_bias):
        self.n_in  = n_in  # кол-во нейронов в входном слое
        self.n_hl  = n_hl  # кол-во нейронов в скрытом слое
        self.n_out = n_out # кол-во нейронов в выходном слое
        # количество скрытых слоёв (минимум 1)
        self.n_hlayers = n_hlayers if n_hlayers > 0 else 1 
        self.w_layers = [] # слои весов (все)
        # случайное назначение весов (вход -> скрытый слой)
        self.w_layers.append(np.random.randn(self.n_in, self.n_hl))
        # случайное назначение весов (скрытый слой -> скрытый слой)
        if self.n_hlayers > 1:
            for _ in range(0, self.n_hlayers - 1):
                self.w_layers.append(np.random.randn(self.n_hl, self.n_hl))
        # случайное назначение весов (скрытый слой -> выход)
        self.w_layers.append(np.random.randn(self.n_hl, self.n_out))
        # случайный веса синапсов, исходящих из 
        # нейронов смещения в скрытых слоях
        self.biases = []
        for _ in range(0, self.n_hlayers):
            self.biases.append( np.random.uniform() if use_bias else 0 )
    
    def forward(self, x):
        "Прямое распространение"
        self.ys = []
        # in -> h
        s_h = np.dot(x, self.w_layers[0])
        y_h = self.sigmoid(s_h)
        self.ys.append(y_h)
        # h -> h -> h
        if self.n_hlayers > 1:
            for i in range(0, self.n_hlayers - 1):
                s_h = np.dot(y_h, self.w_layers[i + 1]) + self.biases[i]
                y_h = self.sigmoid(s_h)
                self.ys.append(y_h)
        # h -> out
        s_h = np.dot(y_h, self.w_layers[-1]) + self.biases[-1]
        return self.sigmoid(s_h)

    def backward(self, x, t, y, alpha, es):
        """
        Обратное распространение (RMS Loss Function)
        :param x - вход
        :param t - ожидаемый результат
        :param y - реальный результат
        :param alpha - момент 
        :param es - скорость обучения
        """
        grads = []
        # ошибка для выходного слоя
        error = t - y
        delta = error * self.sigmoid_prime(y)
        grads.append(self.ys[-1].T.dot(delta))
        # ошибки для скрытых слоёв
        if self.n_hlayers > 1:
            for i in range(self.n_hlayers - 1, 0, -1):
                error = delta.dot(self.w_layers[i+1].T)
                delta = error * self.sigmoid_prime(self.ys[i])
                grads.append(self.ys[i-1].T.dot(delta))
        # ошибка для входного слоя
        error = delta.dot(self.w_layers[1].T)
        delta = error * self.sigmoid_prime(self.ys[0])
        grads.append(x.T.dot(delta))
        # корректировка весов
        s = len(grads)
        for i in range(0, s):
            self.w_layers[i] = es * grads[s-1-i] + alpha * self.w_layers[i] 

    def train(self, x, t, alpha, es):
        """
        Обучение
        :param x - вход
        :param t - ожидаемый результат
        :param alpha - момент 
        :param es - скорость обучения
        """
        y = self.forward(x)                # прямое распространение
        self.backward(x, t, y, alpha, es)  # обратное распространение
    
    def predict(self, x):
        return self.forward(x)

    def sigmoid(self, s):
        "Сигмоидальная логистическая функция активации"
        return 1 / (1 + np.exp(-s))

    def sigmoid_prime(self, s):
        "Производная функции активации"
        return s * (1 - s)

    def save_weights(self, prefix):
        "Сохранение весов в файлы"
        for i, w in enumerate(self.w_layers):
            np.savetxt("{}_{}.w.txt".format(prefix, i), w, fmt="%s")
    
    def load_weights(self, prefix):
        "Загрузка весов из файлов"
        self.w_layers[0] = np.loadtxt("{}_{}.w.txt".format(prefix, 0), dtype=float)
        for i in range(1, len(self.w_layers)-1):
            self.w_layers[i] = np.loadtxt("{}_{}.w.txt".format(prefix, i), dtype=float)
        last = len(self.w_layers)-1
        shape = self.w_layers[-1].shape
        self.w_layers[-1] = np.loadtxt("{}_{}.w.txt".format(prefix, last), dtype=float)
        self.w_layers[-1] = self.w_layers[-1].reshape(shape[0], self.n_out)

    def __str__(self):
        return "\nInputs:  {0}\nOutputs: {1}\nHidden:  {2}\nNeurons in Hidden: {3}\nUse Bias: {4}\n".format(self.n_in, self.n_out, self.n_hlayers, self.n_hl, 'Yes' if self.biases[0] else 'No')
