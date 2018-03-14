# -*- coding: utf-8 -*-
#!/usr/bin/env python

import sys
import numpy as np
from pylab import plt

class Map(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def _route(self, o1, o2, endpoint = False):
        # print "Generating trajectory from {} to {}...".format(o1, o2)
        d = o1[0] - o2[0]
        if not d:
            print "Invalid input"
            sys.exit(1)
        k = (o1[1] - o2[1]) / float(d)
        b = o2[1] - k * o2[0]
        t = [[x, k * x + b] for x in range(o1[0], o2[0])]
        if endpoint:
            t.append(o2)
        return t

    def _generate(self, s, t, d):
        """
        Генерация траектории по трём точкам
        :param s стартовая точка
        :param t промежуточная точка
        :param d цель
        """
        # генерация траектории s-t-d
        t  = self._route(s, t) + self._route(t, d, True)
        return t

    def dataset(self, trajectory, N, uniform = True):
        # Triangle Point Picking
        data = []
        p0 = v0 = np.array(self.o1, dtype=float)
        if trajectory == 0:
            p1 = v1 = np.array(self.a, dtype=float)
            p2 = v2 = np.array(self.o2, dtype=float)
        else:
            p1 = v1 = np.array(self.b, dtype=float)
            p2 = v2 = np.array(self.o2, dtype=float)
        
        # for barycentric coordinates
        area = 0.5 * (-p1[1]*p2[0] + p0[1] * (-p1[0] + p2[0]) + p0[0] * (p1[1] - p2[1]) + p1[0] * p2[1])

        v1 = v1 - v0
        v2 = v2 - v0
        if uniform:
            N *= 2
        for i in xrange(N):
            a1 = np.random.random()
            a2 = np.random.random()
            if uniform:
                x = a1 * v1 + a2 * v2 + v0
                s = 1.0 / (2.0 * area) * (p0[1] * p2[0] - p0[0] * p2[1] + (p2[1] - p0[1]) * x[0] + (p0[0] - p2[0]) * x[1])
                t = 1.0 / (2.0 * area) * (p0[0] * p1[1] - p0[1] * p1[0] + (p0[1] - p1[1]) * x[0] + (p1[0] - p0[0]) * x[1])
                if s > 0 and t > 0 and 1-s-t > 0:
                    data.append(x)
            else:
                x = a1 * v1 + (1 - a1) * a2 * v2 + v0
                data.append(x)
        return np.array(data)
        

    def build(self, dsize):
        # старт
        self.o1 = [0, np.random.randint(self.height-2) + 1]
        # промежуточная точка маршрута 1
        self.a = [np.random.randint(self.width-4) + 2, 0]
        # промежуточная точка маршрута 2
        self.b = [np.random.randint(self.width-4) + 2, self.height-1]
        # цель
        self.o2 = [self.width-1, np.random.randint(self.height-2)+1]
        # генерация траектории o-a-c
        self.t0  = np.array(self._generate(self.o1, self.a, self.o2), dtype=float)
        # генерация траектории o-b-c
        self.t1  = np.array(self._generate(self.o1, self.b, self.o2), dtype=float)
        self.dataset0 = self.dataset(0, dsize)
        self.dataset1 = self.dataset(1, dsize)
        return (self.dataset0, self.dataset1)
    
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
