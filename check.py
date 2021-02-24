import numpy as np
import os
import torch
from matplotlib import pyplot as plt

path = "/home/danai/Desktop/GaitTracking/data/"
paths = ["p1/2.a/", "p5/2.a/", "p11/2.a/", "p16/3.a/", "p17/2.a/", "p17/3.a/", "p18/2.a/", "p18/3.a/"]
for p in paths:
    print(p)
    gaits = np.genfromtxt(path + p + "gait_states.csv", delimiter=",")
    laser = np.genfromtxt(path + p + "laserpoints.csv", delimiter=",")
    valid = open(path + p + "gait_valid.txt", "r")
    for line in valid:
        start, end = line.strip().split(" ")
        i = int(start)
        end = int(end)
        while i <= end:
            print(gaits[i])
            #plt.title(i)
            l1 = laser[i]
            l = l1.reshape((int(l1.shape[0] / 2), 2))
            #v = l[:, 1] < 0.2
            #if i <= 1009:
            #    v += (l[:, 0] > 0.4) + (l[:, 1] < 0.25)
            #else:
            #    v += (l[:, 0] < -0.4) + (l[:, 0] > 0.2) + (l[:, 1] > 0.75)
            #l[v] = 0
            plt.xlim(-0.5, 0.5)
            plt.ylim(0.2, 1.2)
            #rec = plt.Rectangle((-0.5, 0.2), 1, 1, fill=False, linestyle='--', linewidth = 2, ec = 'y')
            #c = plt.Circle((0, 0), 0.1, fill=False, ec='r', lw = 2)
            #ax = plt.arrow(0, 0, 0.1, 0, head_length=0.04, head_width=0.1, color='black', shape='full', ec='black')
            #ay = plt.arrow(0, 0, 0, 0.26, head_length=0.08, head_width=0.04, color='black', shape='full', ec='black')
            #rl = plt.arrow(0, 0, 0, 0.26, head_length=0.08, head_width=0.05, color='black', shape='full', ec='black')
            #ll = plt.arrow(0, 0, 0, 0.26, head_length=0.08, head_width=0.05, color='black', shape='full', ec='black')
            #plt.gca().add_patch(rec)
            #plt.gca().add_patch(c)
            #plt.gca().add_patch(ax)
            #plt.gca().add_patch(ay)
            #plt.text(0.025, 0.3, 'y', size=10, weight='bold')
            #plt.text(0.13, -0.1, 'x', size=10, weight='bold')
            plt.title(i)
            plt.scatter(l[:, 0], l[:, 1], c = 'b', marker = '.')
            plt.show(block=False)
            plt.pause(0.01)
            plt.clf()
            i += 1

#np.savetxt("/home/danai/Desktop/GaitTracking/data/p5/2.a/new_laserpoints.csv", new_laser, delimiter=",")
