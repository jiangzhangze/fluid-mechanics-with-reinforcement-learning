import numpy as np

def probes(x_min, y_min, length,num):
    x = np.linspace(x_min, x_min + length, num)
    y = np.linspace(y_min, y_min + length, num)
    points = np.zeros((3, num))
    points[0] = x
    points[1] = y

    return points