from matplotlib import pyplot as plt
import numpy as np
import os

os.chdir("/home/danai/Desktop/mobot/p1/2.a")
centers = np.genfromtxt("valid_centers.csv", delimiter = ",")
laser = np.genfromtxt("valid_laserpoints.csv", delimiter = ",")
box = [(-0.25, 0.2), (0.22, 1)]

plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.2, 1)
line1, = ax.plot([centers[0, 0], centers[0, 2]], [centers[0, 1], centers[0, 3]], 'o')
line2, = ax.plot(laser[0, range(0,laser.shape[1], 2)], laser[0, range(1,laser.shape[1], 2)], 'o')


for i in range(1, centers.shape[0]):
    line1.set_label(str(i))
    ax.legend()
    line1.set_data([centers[i,0], centers[i,2]], [centers[i,1], centers[i,3]])
    line2.set_data(laser[i][range(0,laser.shape[1], 2)], laser[i][range(1,laser.shape[1], 2)])
    fig.canvas.draw()
    fig.canvas.flush_events()
