import numpy as np
from math import *


def positions(jet_positions):
    jet_coordinates = np.zeros((3, len(jet_positions)))
    jet_x = [i[0] for i in jet_positions]
    jet_y = [i[1] for i in jet_positions]
    jet_coordinates[0] = jet_x
    jet_coordinates[1] = jet_y
    return jet_coordinates


def circle_probes(r, num):
    r = r
    x0 = r * cos(0) + 0.3
    y0 = r * sin(0) + 0.2
    x1 = r * cos(pi / 2) + 0.3
    y1 = r * sin(pi / 2) + 0.2
    x2 = r * cos(3 / 2 * pi) + 0.3
    y2 = r * sin(3 * pi / 2) + 0.2
    x3 = r * cos(pi) + 0.3
    y3 = r * sin(pi) + 0.2
    jet_positions = [(x0, y0, 0), (x1, y1, 0), (x2, y2, 0), (x3, y3, 0)]
    if r == 0.05:
        points = positions(jet_positions)
    else:
        points = np.zeros((3, num))
        for i in range(num):
            angle = 2.0 * pi * float(i / num)
            x = r * cos(angle) + 0.3
            y = r * sin(angle) + 0.2
            points[0, i] = x
            points[1, i] = y
    return points


def probes(x_min, y_min, length, num):
    x = np.linspace(x_min, x_min + length, int(np.sqrt(num)))
    y = np.linspace(y_min, y_min + length, int(np.sqrt(num)))
    xx, yy = np.meshgrid(x, y)
    x = xx.ravel()
    y = yy.ravel()
    points = np.zeros((3, num))
    points[0] = x
    points[1] = y

    return points


def add_points(probes1, probes2):
    points = np.concatenate([probes1, probes2], axis=1)
    return points
