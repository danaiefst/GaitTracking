from matplotlib import pyplot as plt
import numpy as np
import os
import torch
import sys

person = sys.argv[1]
scenario = sys.argv[2]
path = sys.argv[3]

os.chdir("/home/athdom/GaitTracking/{}/{}".format(person, scenario))
valid = open("valid.txt", "r")
centers = np.genfromtxt("valid_centers.csv", delimiter = ",")
laser = np.genfromtxt("valid_laserpoints.csv", delimiter = ",")
max_height = 1.2
min_height = 0.1
max_width = 0.5
min_width = -0.5
img_side = 100
images = []

box = [(-0.25, 0.2), (0.22, 1)]

#plt.ion()

#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.set_xlim(-0.5, 0.5)
#ax.set_ylim(-0.2, 1)
laser_spots = laser[0].reshape((int(laser[0].shape[0] / 2), 2))
in_box1 = np.logical_and(box[0][0] < laser_spots[:, 0], laser_spots[:, 0] < box[1][0])
in_box2 = np.logical_and(box[0][1] < laser_spots[:, 1], laser_spots[:, 1] < box[1][1])
in_box = np.logical_and(in_box1, in_box2)
leg1_dists = np.sqrt(((laser_spots[in_box] - centers[0, :2]) ** 2).sum(axis = 1))
leg2_dists = np.sqrt(((laser_spots[in_box] - centers[0, 2:]) ** 2).sum(axis = 1))
min_dists = np.logical_and(leg1_dists < leg2_dists, laser_spots[in_box][:, 0] > centers[0, 2])
mass_center1 = laser_spots[in_box][np.logical_not(min_dists)].sum(axis = 0) / laser_spots[in_box][np.logical_not(min_dists)].shape[0]
mass_center2 = laser_spots[in_box][min_dists].sum(axis = 0) / laser_spots[in_box][min_dists].shape[0]
prev_center1 = mass_center1
prev_center2 = mass_center2
#line1, = ax.plot([mass_center1[0], mass_center2[0]], [mass_center1[1], mass_center2[1]], 'o')
#line1, = ax.plot([centers[0, 0], centers[0, 2]], [centers[0, 1], centers[0, 3]], 'o')
#line2, = ax.plot(laser_spots[in_box][np.logical_not(min_dists)][:, 0], laser_spots[in_box][np.logical_not(min_dists)][:, 1], 'o')
#line3, = ax.plot(laser_spots[in_box][min_dists][:, 0], laser_spots[in_box][min_dists][:, 1], 'o')
img = np.zeros((img_side, img_side))
y = np.round((laser_spots[in_box][:, 0] - min_width) / (max_width - min_width) * img_side)
x = img_side - np.round((laser_spots[in_box][:, 1] - min_height) / (max_height - min_height) * img_side)
img[x.astype(int), y.astype(int)] = 1
images.append(img)

for i in range(1, centers.shape[0]):
    #print(i)
    img = np.zeros((img_side, img_side))
    laser_spots = laser[i].reshape((int(laser[i].shape[0] / 2), 2))
    in_box1 = np.logical_and(box[0][0] < laser_spots[:, 0], laser_spots[:, 0] < box[1][0])
    in_box2 = np.logical_and(box[0][1] < laser_spots[:, 1], laser_spots[:, 1] < box[1][1])
    in_box = np.logical_and(in_box1, in_box2)
    leg1_dists = np.sqrt(((laser_spots[in_box] - centers[i, :2]) ** 2).sum(axis = 1))
    leg2_dists = np.sqrt(((laser_spots[in_box] - centers[i, 2:]) ** 2).sum(axis = 1))
    min_dists = np.logical_and(leg1_dists < leg2_dists, laser_spots[in_box][:, 0] > centers[i, 2])
    mass_center1 = laser_spots[in_box][np.logical_not(min_dists)].sum(axis = 0) / laser_spots[in_box][np.logical_not(min_dists)].shape[0]
    mass_center2 = laser_spots[in_box][min_dists].sum(axis = 0) / laser_spots[in_box][min_dists].shape[0]
    mass_center1 = (mass_center1 + prev_center1) / 2
    mass_center2 = (mass_center2 + prev_center2) / 2
    #print(mass_center1, mass_center2)
    prev_center1 = mass_center1
    prev_center2 = mass_center2
    #line1.set_data([mass_center1[0], mass_center2[0]], [mass_center1[1], mass_center2[1]])
    #line1.set_data([centers[i, 0], centers[i, 2]], [centers[i, 1], centers[i, 3]])
    #line2.set_data(laser_spots[in_box][np.logical_not(min_dists)][:, 0], laser_spots[in_box][np.logical_not(min_dists)][:, 1])
    #line3.set_data(laser_spots[in_box][min_dists][:, 0], laser_spots[in_box][min_dists][:, 1])
    #fig.canvas.draw()
    #fig.canvas.flush_events()
    y = np.round((laser_spots[in_box][:, 0] - min_width) / (max_width - min_width) * img_side)
    x = img_side - np.round((laser_spots[in_box][:, 1] - min_height) / (max_height - min_height) * img_side)
    #print(x, y, laser_spots[in_box])
    img[x.astype(int), y.astype(int)] = 1
    images.append(img)

data_i = 0
i = 0
for line in valid:
    start, end = line.strip().split(" ")
    i = int(start)
    end = int(end)
    while i <= end:
        torch.save(torch.tensor(images[data_i], dtype=torch.double), "data/{}.pt".format(i))
        data_i += 1
        i += 1
