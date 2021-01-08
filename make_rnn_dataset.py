import numpy as np
import os
import torch
import sys
from data_handler import transformi, transforml
import time

#data_paths=["/gpu-data/athdom/p1/2.a","/gpu-data/athdom/p5/2.a", "/gpu-data/athdom/p11/2.a", "/gpu-data/athdom/p11/3.a", "/gpu-data/athdom/p16/3.a", "/gpu-data/athdom/p17/3.a", "/gpu-data/athdom/p18/3.a", "/gpu-data/athdom/p18/2.a"]
#data_paths1=["/home/athdom/GaitTracking/p1/2.a","/home/athdom/GaitTracking/p5/2.a", "/home/athdom/GaitTracking/p11/2.a", "/home/athdom/GaitTracking/p11/3.a", "/home/athdom/GaitTracking/p16/3.a", "/home/athdom/GaitTracking/p17/3.a", "/home/athdom/GaitTracking/p18/3.a", "/home/athdom/GaitTracking/p18/2.a"]
data_paths=["/home/danai/Desktop/GaitTracking/p1/2.a", "/home/danai/Desktop/GaitTracking/p18/2.a", "/home/danai/Desktop/GaitTracking/p18/3.a"]
data_paths1=["/home/danai/Desktop/GaitTracking/p1/2.a", "/home/danai/Desktop/GaitTracking/p18/2.a", "/home/danai/Desktop/GaitTracking/p18/3.a"]
#data_paths=["/home/shit/Desktop/GaitTracking/p1/2.a","/home/shit/Desktop/GaitTracking/p5/2.a"]#, "/home/shit/Desktop/GaitTracking/p11/2.a", "/home/shit/Desktop/GaitTracking/p11/3.a", "/home/shit/Desktop/GaitTracking/p16/3.a", "/home/shit/Desktop/GaitTracking/p17/3.a", "/home/shit/Desktop/GaitTracking/p18/2.a", "/home/shit/Desktop/GaitTracking/p18/3.a"]
#data_paths1=["/home/shit/Desktop/GaitTracking/p1/2.a","/home/shit/Desktop/GaitTracking/p5/2.a"]#, "/home/shit/Desktop/GaitTracking/p11/2.a", "/home/shit/Desktop/GaitTracking/p11/3.a", "/home/shit/Desktop/GaitTracking/p16/3.a", "/home/shit/Desktop/GaitTracking/p17/3.a", "/home/shit/Desktop/GaitTracking/p18/2.a", "/home/shit/Desktop/GaitTracking/p18/3.a"]
for data_path in range(len(data_paths)):
    print(data_paths[data_path])
    os.chdir(data_paths1[data_path])
    os.system("rm " + data_paths[data_path] + "/data/*")
    os.system("rm " + data_paths[data_path] + "/labels/*")
    valid = open("valid.txt", "r")
    laser = np.genfromtxt("laserpoints.csv", delimiter = ",")
    centers = np.genfromtxt("centers.csv", delimiter = ",")
    items = os.listdir(data_paths[data_path])
    if "labels" not in items:
        os.mkdir(os.path.join(data_paths[data_path], "labels"))
    if "data" not in items:
        os.mkdir(os.path.join(data_paths[data_path], "data"))

    max_height = 1.2
    min_height = 0.1
    max_width = 0.5
    min_width = -0.5
    img_side = 112
    images = []
    tags = []

    box = [(-0.25, 0.2), (0.22, 1)]

    for i in range(laser.shape[0]):
        #t = time.time()
        img = torch.zeros((img_side, img_side), dtype=torch.double)
        laser_spots = laser[i].reshape((int(laser[i].shape[0] / 2), 2))
        in_box1 = np.logical_and(box[0][0] < laser_spots[:, 0], laser_spots[:, 0] < box[1][0])
        in_box2 = np.logical_and(box[0][1] < laser_spots[:, 1], laser_spots[:, 1] < box[1][1])
        in_box = np.logical_and(in_box1, in_box2)
        y = np.round((laser_spots[in_box][:, 0] - min_width) / (max_width - min_width) * img_side)
        x = img_side - np.round((laser_spots[in_box][:, 1] - min_height) / (max_height - min_height) * img_side)
        img[x.astype(int), y.astype(int)] = 1
        images.append(img)

        center = centers[i]
        y1 = (center[0] - min_width) / (max_width - min_width) * img_side
        x1 = img_side - (center[1] - min_height) / (max_height - min_height) * img_side
        y2 = (center[2] - min_width) / (max_width - min_width) * img_side
        x2 = img_side - (center[3] - min_height) / (max_height - min_height) * img_side
        tags.append(torch.tensor([[x1, y1], [x2, y2]]))
        #print(time.time() - t)
    i = 0
    last = laser.shape[0]
    for line in valid:
        start, end = line.strip().split(" ")
        i = int(start)
        end = int(end)
        while i <= end:
            torch.save(images[i], "{}/data/{}.pt".format(data_paths[data_path], i))
            torch.save(tags[i], "{}/labels/{}.pt".format(data_paths[data_path], i))
            if data_path < len(data_paths) - 2:
                transformed = transformi(images[i])
                for j in range(len(transformed)):
                    torch.save(transformed[j], "{}/data/{}.pt".format(data_paths[data_path], j * (last + 1) + i))
                transformed = transforml(tags[i])
                for j in range(len(transformed)):
                    torch.save(transformed[j], "{}/labels/{}.pt".format(data_paths[data_path], j * (last + 1) + i))
            i += 1
