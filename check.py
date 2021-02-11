import numpy as np
import os
import torch
from matplotlib import pyplot as plt

path = "/home/danai/Desktop/GaitTracking/data/p18/3.a"

os.chdir(path)
laser = np.genfromtxt("laserpoints.csv", delimiter=",")
centers = np.genfromtxt("centers.csv", delimiter=",")

max_height = 1.2
min_height = 0.2
max_width = 0.5
min_width = -0.5
img_side = 112

def find_center(label):
    ret = label * img_side
    return ret[0][0], ret[0][1], ret[1][0], ret[1][1]


#plt.show()


for i in range(len(laser)):
    plt.title(i)
    l1 = laser[i]
    l = l1.reshape((int(l1.shape[0] / 2), 2))
    plt.xlim(min_width, max_width)
    plt.ylim(min_height, max_height)
    plt.scatter(l[:, 0], l[:, 1], c = 'b', marker = '.')
    plt.scatter(centers[i][0], centers[i][1], c = 'r', marker = 'v')
    plt.scatter(centers[i][2], centers[i][3], c = 'y', marker = 'v')    
    plt.show(block=False)
    plt.pause(0.01)
    plt.clf()
