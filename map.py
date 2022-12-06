# -*- coding: utf-8 -*-

__author__ = "Ilya Shoshin (Galarius)"

"""
Generation of a conditional map of the terrain, 
two routes and evenly spaced points belonging to one of the two routes.
Visualization of the conditional map of the terrain.
"""
      
import numpy as np
from pylab import plt   

class Map(object):
    """
    Conditional terrain map with two routes.
    Allows to generate evenly distributed points belonging to one of the two routes.
    Provides a visualization of the conditional terrain map.
    """
    def __init__(self, width, height):
        # Map size
        self.width = width
        self.height = height
        # Start
        self.o1 = np.array([0, np.random.randint(self.height-2) + 1], dtype=float)
        # Intermediate point 1 of route
        self.a = np.array([np.random.randint(self.width-4) + 2, 0], dtype=float)
        # Intermediate point 2 of route
        self.b = np.array([np.random.randint(self.width-4) + 2, self.height], dtype=float)
        # Destination
        self.o2 = np.array([self.width, np.random.randint(self.height-2)+1], dtype=float)

    def dataset(self, trajectory, npoints, uniform = True):
        # Triangle Point Picking
        data = []
        if trajectory == 0:
            data = Map.distrInTriangle(self.o1, self.a, self.o2, npoints * 2, uniform)
            p = np.array([0, 0], dtype = float)
            data.extend(Map.distrInTriangle(self.o1, p, self.a, npoints, uniform))
            p = np.array([self.width, 0], dtype = float)
            data.extend(Map.distrInTriangle(self.a, p, self.o2, npoints, uniform))
        else:
            data = Map.distrInTriangle(self.o1, self.b, self.o2, npoints * 2, uniform)
            p = np.array([0, self.height], dtype = float)
            data.extend(Map.distrInTriangle(p, self.o1, self.b, npoints, uniform))
            p = np.array([self.width, self.height], dtype = float)
            data.extend(Map.distrInTriangle(self.b, self.o2, p, npoints, uniform))
        return np.array(data)

    def plotMap(self, fname=None):
        _, ax = plt.subplots()
        ax.plot([self.o1[0], self.a[0], self.o2[0]], [self.o1[1], self.a[1], self.o2[1]], 'r', label='Trajectory 0')
        ax.plot([self.o1[0], self.b[0], self.o2[0]], [self.o1[1], self.b[1], self.o2[1]], 'b--', label='Trajectory 1')
        plt.title("Map")
        plt.grid(True)
        if fname:
            plt.savefig(fname)
        plt.show()

    def plot(self, good, bad, dataset0, dataset1, fname=None):
        _, ax = plt.subplots()
        ax.plot([self.o1[0], self.a[0], self.o2[0]], [self.o1[1], self.a[1], self.o2[1]], 'r', label='Trajectory 0')
        ax.plot([self.o1[0], self.b[0], self.o2[0]], [self.o1[1], self.b[1], self.o2[1]], 'b--', label='Trajectory 1')
        if dataset0.any():
            ax.plot(dataset0[:,0], dataset0[:,1], 'ro', label='Train Dataset 0')
        if dataset1.any():
            ax.plot(dataset1[:,0], dataset1[:,1], 'b*', label='Train Dataset 1')
        if good.any():
            ax.plot(good[:,0], good[:,1], 'go', markersize=10, label='Correct prediction')
        if bad.any():
            ax.plot(bad[:,0], bad[:,1], 'black', linestyle='none', marker='D', markersize=10, label='Incorrect prediction')
        plt.title("Map")
        plt.grid(True)
        if fname:
            plt.savefig(fname)
        plt.show()

    @staticmethod
    def triangleArea(p0, p1, p2):
        """
        Calculates the area of a triangle.
        :param p0,p1,p2 - triangle coordinates
        """
        return 0.5 * (-p1[1] * p2[0] + p0[1] * (-p1[0] + p2[0]) + p0[0] * (p1[1] - p2[1]) + p1[0] * p2[1])

    @staticmethod
    def insideTriangle(p, p0, p1, p2, area):
        """
        Checks for a point inside a triangle using barycentric coordinates.
        :param p - coordinates of the point to check
        :param p0,p1,p2 - triangle coordinates
        """
        s = 1.0 / (2.0 * area) * (p0[1] * p2[0] - p0[0] * p2[1] + (p2[1] - p0[1]) * p[0] + (p0[0] - p2[0]) * p[1])
        t = 1.0 / (2.0 * area) * (p0[0] * p1[1] - p0[1] * p1[0] + (p0[1] - p1[1]) * p[0] + (p1[0] - p0[0]) * p[1])
        return s > 0 and t > 0 and 1-s-t > 0

    @staticmethod
    def distrInTriangle(p0, p1, p2, npoints, uniform = True):
        """
        Generates points inside a triangle by the Triangle Point Picking method.
        :param p0,p1,p2 - triangle coordinates
        :param npoints - number of points
        :uniform - even distribution
        """
        data = []
        v0 = p0
        v1 = p1 - v0
        v2 = p2 - v0
        area = Map.triangleArea(p0, p1, p2)
        if uniform:
            npoints *= 2
        for _ in range(npoints):
            a1 = np.random.random()
            a2 = np.random.random()
            if uniform:
                x = a1 * v1 + a2 * v2 + v0
                if Map.insideTriangle(x, p0, p1, p2, area):
                    data.append(x)
            else:
                x = a1 * v1 + (1 - a1) * a2 * v2 + v0
                data.append(x)
        return data
