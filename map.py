# -*- coding: utf-8 -*-
#!/usr/bin/env python

__author__ = "Ilya Shoshin"
__copyright__   = "(c) MAI, M30-102-17"

"""
Генерация условной карты местности, 
двух маршрутов и равномерно распределённых
точек, относящихся к одному из двух маршрутов.
Визуализация условной карты местности.
"""

import sys              # exit()
import numpy as np      # матрицы и вектора
from pylab import plt   # графики

def triangleArea(p0, p1, p2):
    """
    Вычисление площади треугольника
    :param p0,p1,p2 - координаты треугольника
    """
    return 0.5 * (-p1[1] * p2[0] + p0[1] * (-p1[0] + p2[0]) + p0[0] * (p1[1] - p2[1]) + p1[0] * p2[1])

def insideTriangle(p, p0, p1, p2, area):
    """
    Проверка нахождения точки внутри треугольника 
    при помощи барицентрических координат.
    :param p - координаты точки для проверки
    :param p0,p1,p2 - координаты треугольника
    """
    s = 1.0 / (2.0 * area) * (p0[1] * p2[0] - p0[0] * p2[1] + (p2[1] - p0[1]) * p[0] + (p0[0] - p2[0]) * p[1])
    t = 1.0 / (2.0 * area) * (p0[0] * p1[1] - p0[1] * p1[0] + (p0[1] - p1[1]) * p[0] + (p1[0] - p0[0]) * p[1])
    return s > 0 and t > 0 and 1-s-t > 0

def distrInTriangle(p0, p1, p2, npoints, uniform = True):
    """
    Генерация точек внутри треугольника 
    методом Triangle Point Picking
    :param p0,p1,p2 - координаты треугольника
    :param npoints - количество точек
    :uniform - равномерное распределение
    """
    data = []
    v0 = p0
    v1 = p1 - v0
    v2 = p2 - v0
    area = triangleArea(p0, p1, p2)
    if uniform:
        npoints *= 2
    for i in xrange(npoints):
        a1 = np.random.random()
        a2 = np.random.random()
        if uniform:
            x = a1 * v1 + a2 * v2 + v0
            if insideTriangle(x, p0, p1, p2, area):
                data.append(x)
        else:
            x = a1 * v1 + (1 - a1) * a2 * v2 + v0
            data.append(x)
    return data

class Map(object):
    """
    Условная карта местности с двумя маршрутами.
    Позволяет сгенерировать равномерно распределённые
    точки, относящиеся к одному из двух маршрутов.
    Предоставляет визуализацию условной карты местности.
    """
    def __init__(self, width, height):
        # размер карты
        self.width = width
        self.height = height
        # старт
        self.o1 = np.array([0, np.random.randint(self.height-2) + 1], dtype=float)
        # промежуточная точка маршрута 1
        self.a = np.array([np.random.randint(self.width-4) + 2, 0], dtype=float)
        # промежуточная точка маршрута 2
        self.b = np.array([np.random.randint(self.width-4) + 2, self.height], dtype=float)
        # цель
        self.o2 = np.array([self.width, np.random.randint(self.height-2)+1], dtype=float)
        # площадь треугольника соответствующего 1ой траектории
        self.area0 = triangleArea(self.o1, self.a, self.o2)
        # площадь треугольника соответствующего 2ой траектории
        self.area2 = triangleArea(self.o1, self.b, self.o2)

    def _route(self, o1, o2, endpoint = False):
        """
        Генерация набора координат, принадлежащих
        траектории с помощью уравнения прямой:
            y = k * x + b
        :param o1 - координаты начала прямого участка траектории
        :param o2 - координаты конца прямого участка траектории
        """
        # вычисление коэффициентов прямой
        d = o1[0] - o2[0]
        if not d:
            print "Invalid input."
            sys.exit(1)
        k = (o1[1] - o2[1]) / float(d)
        b = o2[1] - k * o2[0]
        # генерация точек
        s, e = o1.astype(int)[0], o2.astype(int)[0]
        t = [[x, k * x + b] for x in range(s, e)]
        if endpoint:    
            # добавление в список последней точки
            t.append(o2)
        return t

    def _generate(self, s, t, d):
        """
        Генерация траектории по трём точкам
        :param s стартовая точка
        :param t промежуточная точка
        :param d цель
        """
        # генерация траектории s-t-d, состоящей из двух прямых
        t  = self._route(s, t) + self._route(t, d, True)
        return t
    
    def dataset(self, trajectory, npoints, uniform = True):
        # Triangle Point Picking
        data = []
        if trajectory == 0:
            data = distrInTriangle(self.o1, self.a, self.o2, npoints, uniform)
            p = np.array([0, 0], dtype = float)
            data.extend(distrInTriangle(self.o1, p, self.a, npoints, uniform))
            p = np.array([self.width, 0], dtype = float)
            data.extend(distrInTriangle(self.a, p, self.o2, npoints, uniform))
        else:
            data = distrInTriangle(self.o1, self.b, self.o2, npoints, uniform)
            p = np.array([0, self.height], dtype = float)
            data.extend(distrInTriangle(p, self.o1, self.b, npoints, uniform))
            p = np.array([self.width, self.height], dtype = float)
            data.extend(distrInTriangle(self.b, self.o2, p, npoints, uniform))
        return np.array(data)
        
    def build(self, dsize):
        # генерация траектории o-a-c
        self.t0  = np.array(self._generate(self.o1, self.a, self.o2), dtype=float)
        # генерация траектории o-b-c
        self.t1  = np.array(self._generate(self.o1, self.b, self.o2), dtype=float)
        self.dataset0 = self.dataset(0, dsize)
        self.dataset1 = self.dataset(1, dsize)
        return (self.dataset0, self.dataset1)
    
    def plotTrajectories(self, fname=None):
        fig, ax = plt.subplots()
        ax.plot(self.t0[:,0], self.t0[:,1], 'r', label='Trajectory 0')
        ax.plot(self.t1[:,0], self.t1[:,1], 'b', label='Trajectory 1')
        legend = ax.legend(loc='best', framealpha=0.5)
        plt.title("Map")
        plt.grid(True)
        if fname:
            plt.savefig(fname)
        plt.show()

    def plot(self, predict, save=False):
        fig, ax = plt.subplots()
        ax.plot(self.t0[:,0], self.t0[:,1], 'r', label='Trajectory 0')
        ax.plot(self.t1[:,0], self.t1[:,1], 'b', label='Trajectory 1')
        ax.plot(self.dataset0[:,0], self.dataset0[:,1], 'ro', label='Train Dataset 0')
        ax.plot(self.dataset1[:,0], self.dataset1[:,1], 'bo', label='Train Dataset 1')
        ax.plot(predict[:,0], predict[:,1], 'go', markersize=10, label='Dataset to Predict')
        legend = ax.legend(loc='best', framealpha=0.5)
        plt.title("Map")
        plt.grid(True)
        if save:
            plt.savefig('map.png')
        plt.show()
