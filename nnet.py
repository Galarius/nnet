# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np

class NeuralNetwork(object):
    def __init__(self, n_in, n_hl, n_out, use_bias = False):
        self.n_in  = n_in  # кол-во нейронов в входном слое
        self.n_hl  = n_hl  # кол-во нейронов в скрытом слое
        self.n_out = n_out # кол-во нейронов в выходном слое
        # случайное назначение весов (вход -> скрытый слой)
        self.w_in_h = np.random.randn(self.n_in, self.n_hl)
        # случайное назначение весов (скрытый слой -> выход)
        self.w_h_out = np.random.randn(self.n_hl, self.n_out)
        # случайный вес синопса, исходящего из 
        # нейрона смещения в скрытом слое
        self.bias = use_bias if np.random.uniform() else 0
    
    def forward(self, x):
        "Прямое распространение"
        # скалярное произведение входных данных и весов
        s_h = np.dot(x, self.w_in_h) 
        # активация нейронов скрытого слоя
        self.y_h = self.sigmoid(s_h) 
        # скалярное произведение активированных нейронов
        # скрытого слоя и весов + вес синопса от нейрона смещения
        s_o = np.dot(self.y_h, self.w_h_out) + self.bias
        # активация нейронов выходного слоя
        return self.sigmoid(s_o)

    def backward(self, x, t, y, e = 1, a = 1):
        """
        Обратное распространение (RMS Loss Function)
        :param x - вход
        :param t - ожидаемый результат
        :param y - реальный результат
        :param e - скорость обучения
        :param a - момент 
        """
        o_error = t - y  # ошибка для выходного слоя
        o_delta = o_error * self.sigmoid_prime(y)
        o_grad = self.y_h.T.dot(o_delta)
        # ошибка для скрытого слоя
        h_error = o_delta.dot(self.w_h_out.T)
        h_delta = h_error * self.sigmoid_prime(self.y_h)
        h_grad = x.T.dot(h_delta)
        # изменение весов
        self.w_in_h = e * h_grad + a * self.w_in_h
        self.w_h_out = e * o_grad + a * self.w_h_out

    def train(self, x, t):
        "Обучение"
        o = self.forward(x)     # прямое распространение
        self.backward(x, t, o)  # обратное распространение
    
    def predict(self, x):
        return self.forward(x)

    def sigmoid(self, s):
        "Сигмоидальная логистическая функция активации"
        return 1 / (1 + np.exp(-s))

    def sigmoid_prime(self, s):
        "Производная функции активации"
        return s * (1 - s)

    def save_weights(self, f_w_in_h, f_w_h_out):
        "Сохранение весов в файл"
        np.savetxt(f_w_in_h, self.w_in_h, fmt="%s")
        np.savetxt(f_w_h_out, self.w_h_out, fmt="%s")
    
    def load_weights(self, f_w_in_h, f_w_h_out):
        "Загрузка весов из файла"
        self.w_in_h = np.loadtxt(f_w_in_h, f_w_h_out, dtype=float)
        self.w_h_out = np.loadtxt(f_w_h_out, dtype=float)
        self.w_h_out = self.w_h_out.reshape(self.w_h_out.shape[0], 1)
        
