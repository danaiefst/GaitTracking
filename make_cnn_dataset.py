import numpy as np
import os
import torch
import sys
from data_handler import print_data
from matplotlib import pyplot as plt

#data_paths=["/gpu-data/athdom/p1/2.a","/gpu-data/athdom/p5/2.a", "/gpu-data/athdom/p11/2.a", "/gpu-data/athdom/p11/3.a", "/gpu-data/athdom/p16/3.a", "/gpu-data/athdom/p17/3.a", "/gpu-data/athdom/p18/2.a", "/gpu-data/athdom/p18/3.a"]
#data_paths1=["/home/athdom/GaitTracking/p1/2.a","/home/athdom/GaitTracking/p5/2.a", "/home/athdom/GaitTracking/p11/2.a", "/home/athdom/GaitTracking/p11/3.a", "/home/athdom/GaitTracking/p16/3.a", "/home/athdom/GaitTracking/p17/3.a", "/home/athdom/GaitTracking/p18/2.a", "/home/athdom/GaitTracking/p18/3.a"]
data_paths=["/home/danai/Desktop/GaitTracking/p1/2.a"]
data_paths1=["/home/danai/Desktop/GaitTracking/p1/2.a"]
#data_paths=["/home/shit/Desktop/GaitTracking/p1/2.a","/home/shit/Desktop/GaitTracking/p5/2.a"]#, "/home/shit/Desktop/GaitTracking/p11/2.a", "/home/shit/Desktop/GaitTracking/p11/3.a", "/home/shit/Desktop/GaitTracking/p16/3.a", "/home/shit/Desktop/GaitTracking/p17/3.a", "/home/shit/Desktop/GaitTracking/p18/2.a", "/home/shit/Desktop/GaitTracking/p18/3.a"]
#data_paths1=["/home/shit/Desktop/GaitTracking/p1/2.a","/home/shit/Desktop/GaitTracking/p5/2.a"]#, "/home/shit/Desktop/GaitTracking/p11/2.a", "/home/shit/Desktop/GaitTracking/p11/3.a", "/home/shit/Desktop/GaitTracking/p16/3.a", "/home/shit/Desktop/GaitTracking/p17/3.a", "/home/shit/Desktop/GaitTracking/p18/2.a", "/home/shit/Desktop/GaitTracking/p18/3.a"]

for data_path in range(len(data_paths)):
    os.chdir(data_paths1[data_path])
    valid = open("valid.txt", "r")
    laser = np.genfromtxt("laserpoints.csv", delimiter = ",")
    centers = np.genfromtxt("centers.csv", delimiter = ",")

    max_height = 1.2
    min_height = 0.1
    max_width = 0.5
    min_width = -0.5
    img_side = 112
    grid = 7
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
        y = np.round((laser_spots[in_box][:, 0] - min_width) / (max_width - min_width) * img_side)
        x = img_side - np.round((laser_spots[in_box][:, 1] - min_height) / (max_height - min_height) * img_side)

        center = centers[i]
        y1 = int((center[0] - min_width) / (max_width - min_width) * img_side)
        x1 = int(img_side - (center[1] - min_height) / (max_height - min_height) * img_side)
        y2 = int((center[2] - min_width) / (max_width - min_width) * img_side)
        x2 = int(img_side - (center[3] - min_height) / (max_height - min_height) * img_side)

        if 0 <= x1 < img_side and 0 <= y1 < img_side and 0 <= x2 < img_side and 0 <= y2 < img_side:
            dists1 = (x - x1) ** 2 + (y - y1) ** 2
            dists2 = (x - x2) ** 2 + (y - y2) ** 2
            r = dists2 >= dists1
            l = dists1 > dists2
            points_r.append((x[r].astype(int), y[r].astype(int), x1, y1))
            points_l.append((x[l].astype(int), y[l].astype(int), x2, y2))
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
               img[points_r[i][0], points_r[i][1]] = 1
               img[points_l[i][0], points_l[i][1]] = 1
               tag = torch.tensor([[int(points_r[i][2] / img_side * grid), int(points_r[i][3] / img_side * grid), (points_r[i][2] / img_side * grid) % 1, (points_r[i][3] / img_side * grid) % 1], [int(points_l[i][2] / img_side * grid), int(points_l[i][3] / img_side * grid), (points_l[i][2] / img_side * grid) % 1, (points_l[i][3] / img_side * grid) % 1]], dtype=torch.double)
               torch.save(img, "{}/data_cnn/{}.pt".format(data_paths[data_path], file))
               torch.save(tag, "{}/labels_cnn/{}.pt".format(data_paths[data_path], file))
               #print_data(img, tag)
               file += 1
            i += 1
    print(file)
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
            leg1_x = (new_points_r[r[i]][0] + x_r[j] * img_side - new_points_r[r[i]][2]).astype(int)
            leg1_y = (new_points_r[r[i]][1] + y_r[j] * img_side - new_points_r[r[i]][3]).astype(int)
            leg2_x = (new_points_l[l[i]][0] + x_l[j] * img_side - new_points_l[l[i]][2]).astype(int)
            leg2_y = (new_points_l[l[i]][1] + y_l[j] * img_side - new_points_l[l[i]][3]).astype(int)
            valid_leg1 = (leg1_x >= 0) * (leg1_x < img_side) * (leg1_y >= 0) * (leg1_y < img_side)
            valid_leg2 = (leg2_x >= 0) * (leg2_x < img_side) * (leg2_y >= 0) * (leg2_y < img_side)
            img[leg1_x[valid_leg1], leg1_y[valid_leg1]] = 1
            img[leg2_x[valid_leg2], leg2_y[valid_leg2]] = 1
            label1 = [int(x_r[j] * grid), int(y_r[j] * grid), (x_r[j] * grid) % 1, (y_r[j] * grid) % 1]
            label2 = [int(x_l[j] * grid), int(y_l[j] * grid), (x_l[j] * grid) % 1, (y_l[j] * grid) % 1]
            if y_r[j] > y_l[j]:
                tag = torch.tensor([label2, label1], dtype=torch.double)
            else:
                tag = torch.tensor([label1, label2], dtype=torch.double)
            torch.save(img, "{}/data_cnn/{}.pt".format(data_paths[data_path], file))
            torch.save(tag, "{}/labels_cnn/{}.pt".format(data_paths[data_path], file))
            print_data(img, tag)
            file += 1
