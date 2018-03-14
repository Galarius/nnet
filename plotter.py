# -*- coding: utf-8 -*-
#!/usr/bin/env python

import os
import numpy as np
from pylab import plt

# T0 = np.array((
# [0, 6], [1, 5], [2, 4], [3, 3], [4,2], [5,1], [6,0], 

# ), dtype=float) 

def plot_map(xs, ys):
    plt.plot(xs, ys)
    plt.grid(True)
    plt.show()