import numpy as np
import os
import torch
import sys
from data_handler import print_data
from matplotlib import pyplot as plt

#data_paths=["/gpu-data/athdom/p1/2.a","/gpu-data/athdom/p5/2.a", "/gpu-data/athdom/p11/2.a", "/gpu-data/athdom/p11/3.a", "/gpu-data/athdom/p16/3.a", "/gpu-data/athdom/p17/3.a", "/gpu-data/athdom/p18/3.a", "/gpu-data/athdom/p18/2.a"]
#data_paths1=["/home/athdom/GaitTracking/p1/2.a","/home/athdom/GaitTracking/p5/2.a", "/home/athdom/GaitTracking/p11/2.a", "/home/athdom/GaitTracking/p11/3.a", "/home/athdom/GaitTracking/p16/3.a", "/home/athdom/GaitTracking/p17/3.a", "/home/athdom/GaitTracking/p18/3.a", "/home/athdom/GaitTracking/p18/2.a"]
#data_paths=["/gpu-data/athdom/p17/2.a"]
#data_paths1=["/home/athdom/GaitTracking/p17/2.a"]
#data_paths=["/home/shit/Desktop/GaitTracking/p1/2.a","/home/shit/Desktop/GaitTracking/p5/2.a"]#, "/home/shit/Desktop/GaitTracking/p11/2.a", "/home/shit/Desktop/GaitTracking/p11/3.a", "/home/shit/Desktop/GaitTracking/p16/3.a", "/home/shit/Desktop/GaitTracking/p17/3.a", "/home/shit/Desktop/GaitTracking/p18/2.a", "/home/shit/Desktop/GaitTracking/p18/3.a"]
#data_paths1=["/home/shit/Desktop/GaitTracking/p1/2.a","/home/shit/Desktop/GaitTracking/p5/2.a"]#, "/home/shit/Desktop/GaitTracking/p11/2.a", "/home/shit/Desktop/GaitTracking/p11/3.a", "/home/shit/Desktop/GaitTracking/p16/3.a", "/home/shit/Desktop/GaitTracking/p17/3.a", "/home/shit/Desktop/GaitTracking/p18/2.a", "/home/shit/Desktop/GaitTracking/p18/3.a"]
data_paths=["/home/danai/Desktop/GaitTracking/p1/2.a","/home/danai/Desktop/GaitTracking/p5/2.a", "/home/danai/Desktop/GaitTracking/p11/2.a", "/home/danai/Desktop/GaitTracking/p11/3.a", "/home/danai/Desktop/GaitTracking/p16/3.a", "/home/danai/Desktop/GaitTracking/p17/2.a", "/home/danai/Desktop/GaitTracking/p17/3.a", "/home/danai/Desktop/GaitTracking/p18/2.a", "/home/danai/Desktop/GaitTracking/p18/3.a"]
data_paths1=["/home/danai/Desktop/GaitTracking/p1/2.a","/home/danai/Desktop/GaitTracking/p5/2.a", "/home/danai/Desktop/GaitTracking/p11/2.a", "/home/danai/Desktop/GaitTracking/p11/3.a", "/home/danai/Desktop/GaitTracking/p16/3.a", "/home/danai/Desktop/GaitTracking/p17/2.a", "/home/danai/Desktop/GaitTracking/p17/3.a", "/home/danai/Desktop/GaitTracking/p18/2.a", "/home/danai/Desktop/GaitTracking/p18/3.a"]


for data_path in range(len(data_paths)):
    print(data_paths[data_path])
    os.chdir(data_paths1[data_path])
    os.system("rm " + data_paths[data_path] + "/data_cnn/*")
    os.system("rm " + data_paths[data_path] + "/labels_cnn/*")
    valid = open("valid.txt", "r")
    laser = np.genfromtxt("laserpoints.csv", delimiter = ",")
    centers = np.genfromtxt("centers.csv", delimiter = ",")
    items = os.listdir(data_paths[data_path])
    if "labels_cnn" not in items:
        os.mkdir(os.path.join(data_paths[data_path], "labels_cnn"))
    if "data_cnn" not in items:
        os.mkdir(os.path.join(data_paths[data_path], "data_cnn"))
    max_height = 1.2
    min_height = 0.1
    max_width = 0.5
    min_width = -0.5
    img_side = 112
    points_r = []
    points_l = []
    images = []
    box = [(-0.25, 0.2), (0.22, 1)]
    file = 0
    for i in range(laser.shape[0]):
        img = np.zeros((img_side, img_side)).astype(np.uint8)
        laser_spots = laser[i].reshape((int(laser[i].shape[0] / 2), 2))
        in_box1 = np.logical_and(box[0][0] < laser_spots[:, 0], laser_spots[:, 0] < box[1][0])
        in_box2 = np.logical_and(box[0][1] < laser_spots[:, 1], laser_spots[:, 1] < box[1][1])
        in_box = np.logical_and(in_box1, in_box2)
        y = (laser_spots[in_box][:, 0] - min_width) / (max_width - min_width)
        x = 1 - (laser_spots[in_box][:, 1] - min_height) / (max_height - min_height)

        center = centers[i]
        y1 = (center[0] - min_width) / (max_width - min_width)
        x1 = 1 - (center[1] - min_height) / (max_height - min_height)
        y2 = (center[2] - min_width) / (max_width - min_width)
        x2 = 1 - (center[3] - min_height) / (max_height - min_height)

        if 0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1:
            dists1 = (x - x1) ** 2 + (y - y1) ** 2
            dists2 = (x - x2) ** 2 + (y - y2) ** 2
            r = dists2 >= dists1
            l = dists1 > dists2
            points_r.append((x[r], y[r], x1, y1))
            points_l.append((x[l], y[l], x2, y2))
        else:
            points_r.append(())
            points_l.append(())
    i = 0

    new_points_r = []
    new_points_l = []
    for line in valid:
        start, end = line.strip().split(" ")
        i = int(start)
        end = int(end)
        while i <= end:
            if points_r[i] == ():
                print("skatoules")
            else:
               new_points_r.append(points_r[i])
               new_points_l.append(points_l[i])
               img = torch.zeros((img_side, img_side), dtype=torch.double)
               img[(points_r[i][0] * (img_side - 1)).astype(int), (points_r[i][1] * (img_side - 1)).astype(int)] = 1
               img[(points_l[i][0] * (img_side - 1)).astype(int), (points_l[i][1] * (img_side - 1)).astype(int)] = 1
               tag = torch.tensor([[points_r[i][2], points_r[i][3]], [points_l[i][2], points_l[i][3]]], dtype=torch.double)
               torch.save(img, "{}/data_cnn/{}.pt".format(data_paths[data_path], file))
               torch.save(tag, "{}/labels_cnn/{}.pt".format(data_paths[data_path], file))
               file += 1
            i += 1
    if data_path < len(data_paths[:-2]):
        np.random.seed(0)
        r = np.random.randint(0, len(new_points_r) - 1, 1000)
        l = np.random.randint(0, len(new_points_l) - 1, 1000)
        x_l = np.random.random(10)
        y_l = np.random.random(10)
        x_r = np.random.random(10)
        y_r = np.random.random(10)
        for i in range(len(r)):
            for j in range(10):
                img = torch.zeros((img_side, img_side), dtype=torch.double)
                leg1_x = new_points_r[r[i]][0] + x_r[j] - new_points_r[r[i]][2]
                leg1_y = new_points_r[r[i]][1] + y_r[j] - new_points_r[r[i]][3]
                leg2_x = new_points_l[l[i]][0] + x_l[j] - new_points_l[l[i]][2]
                leg2_y = new_points_l[l[i]][1] + y_l[j] - new_points_l[l[i]][3]
                valid_leg1 = (leg1_x >= 0) * (leg1_x <= 1) * (leg1_y >= 0) * (leg1_y <= 1)
                valid_leg2 = (leg2_x >= 0) * (leg2_x <= 1) * (leg2_y >= 0) * (leg2_y <= 1)
                img[(leg1_x[valid_leg1] * (img_side - 1)).astype(int), (leg1_y[valid_leg1] * (img_side - 1)).astype(int)] = 1
                img[(leg2_x[valid_leg2] * (img_side - 1)).astype(int), (leg2_y[valid_leg2] * (img_side - 1)).astype(int)] = 1
                label1 = [x_r[j], y_r[j]]
                label2 = [x_l[j], y_l[j]]
                if y_r[j] > y_l[j]:
                    tag = torch.tensor([label2, label1], dtype=torch.double)
                else:
                    tag = torch.tensor([label1, label2], dtype=torch.double)
                    torch.save(img, "{}/data_cnn/{}.pt".format(data_paths[data_path], file))
                    torch.save(tag, "{}/labels_cnn/{}.pt".format(data_paths[data_path], file))
                    file += 1
