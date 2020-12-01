from matplotlib import pyplot as plt
import numpy as np
import os
import torch
import sys
from data_handler import transformi

person = sys.argv[1]
scenario = sys.argv[2]
source_path = sys.argv[3]
dest_path = sys.argv[4]

os.chdir("{}{}/{}".format(source_path, person, scenario))
valid = open("valid.txt", "r")
laser = np.genfromtxt("laserpoints.csv", delimiter = ",")
max_height = 1.2
min_height = 0.1
max_width = 0.5
min_width = -0.5
img_side = 100
images = []

box = [(-0.25, 0.2), (0.22, 1)]

for i in range(laser.shape[0]):
    img = np.zeros((img_side, img_side))
    laser_spots = laser[i].reshape((int(laser[i].shape[0] / 2), 2))
    in_box1 = np.logical_and(box[0][0] < laser_spots[:, 0], laser_spots[:, 0] < box[1][0])
    in_box2 = np.logical_and(box[0][1] < laser_spots[:, 1], laser_spots[:, 1] < box[1][1])
    in_box = np.logical_and(in_box1, in_box2)
    y = np.round((laser_spots[in_box][:, 0] - min_width) / (max_width - min_width) * img_side)
    x = img_side - np.round((laser_spots[in_box][:, 1] - min_height) / (max_height - min_height) * img_side)
    img[x.astype(int), y.astype(int)] = 1
    images.append(img)

i = 0
last = laser.shape[0]
for line in valid:
    start, end = line.strip().split(" ")
    i = int(start)
    end = int(end)
    while i <= end:
        torch.save(torch.tensor(images[i], dtype=torch.double), "{}{}/{}/data/{}.pt".format(dest_path, person, scenario, i))
        transformed = transformi(torch.tensor(images[i], dtype=torch.double))
        for j in range(len(transformed)):
            torch.save(transformed[j], "{}{}/{}/data/{}.pt".format(dest_path, person, scenario, j * (last + 1) + i))
        i += 1
