import numpy as np
import os
import torch
from matplotlib import pyplot as plt

path = "/home/danai/Downloads/"

os.chdir(path)
gaits = np.genfromtxt("/home/danai/Desktop/data/p5/2.a/gait_states.csv", delimiter=",")
laser = np.genfromtxt("/home/danai/Desktop/GaitTracking/data/p5/2.a/new_laserpoints.csv", delimiter=",")
#centers = np.genfromtxt("/home/danai/Desktop/GaitTracking/data/p5/2.a/centers.csv", delimiter=",")
max_height = 1.2
min_height = 0.2
max_width = 0.5
min_width = -0.5
img_side = 112
#plt.show()
new_laser = laser.copy()

for i in range(999, len(laser)):
    print(gaits[i])
    plt.title(i)
    l1 = laser[i]
    l = l1.reshape((int(l1.shape[0] / 2), 2))
    #v = l[:, 1] < 0.2
    #if i <= 1009:
    #    v += (l[:, 0] > 0.4) + (l[:, 1] < 0.25)
    #else:
    #    v += (l[:, 0] < -0.4) + (l[:, 0] > 0.2) + (l[:, 1] > 0.75)
    #l[v] = 0
    new_laser[i] = l.reshape(new_laser[i].shape)
    plt.xlim(min_width, max_width)
    plt.ylim(min_height, max_height)
    plt.scatter(l[:, 0], l[:, 1], c = 'b', marker = '.')
    plt.show(block=False)
    plt.pause(0.01)
    plt.clf()

#np.savetxt("/home/danai/Desktop/GaitTracking/data/p5/2.a/new_laserpoints.csv", new_laser, delimiter=",")
