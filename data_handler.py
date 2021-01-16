import torch
import os
from matplotlib import pyplot as plt
import numpy as np

max_height = 1.2
min_height = 0.2
max_width = 0.5
min_width = -0.5
img_side = 112
grid = 7
box = [(-0.25, 0.2), (0.22, 1)]
shifts = torch.tensor([[-15, 10], [-10, -10], [-10, 10], [-5, 10], [-7, 5], [0, 10]], dtype=torch.double)
torch.set_default_dtype(torch.double)


def mirrori(img):
    new_img = img.flip(-1)
    return new_img

def mirrorl(labels):
    new_labels = torch.tensor([[labels[1, 0], 1 - labels[1, 1]], [labels[0, 0], 1 - labels[0, 1]]])
    return new_labels

def shifti(img, x, y):
    new_img = torch.zeros(img.shape, dtype=torch.double)
    if x >= 0 and y >= 0:
        new_img[x:, y:] = img[:img.shape[0] - x, :img.shape[1] - y]
    elif x <= 0 and y >= 0:
        new_img[:img.shape[0] + x, y:] = img[-x:, :img.shape[1] - y]
    elif x >= 0 and y <= 0:
        new_img[x:, :img.shape[1] + y] = img[:img.shape[0] - x, -y:]
    else:
        new_img[:img.shape[0] + x, :img.shape[1] + y] = img[-x:, -y:]
    return new_img

def print_data(img, labels, fast=0):
    image = img.detach().clone()
    x1, y1, x2, y2 = int(labels[0, 0]), int(labels[0, 1]), int(labels[1, 0]), int(labels[1, 1])
    image[x1, y1] = 0.2
    image[x2, y2] = 0.8
    plt.imshow(image)
    if fast:
        plt.show(block=False)
        plt.pause(fast)
        plt.clf()
    else:
        plt.show()

def find_center(label):
    ret = label / grid * img_side
    return ret[0], ret[1], ret[2], ret[3]

class LegDataLoader():

    """expecting to find at data_paths a data and a labels folder"""
    def __init__(self, batch_size = 32, grid = 7, cnn = 0, data_paths=["/gpu-data/athdom/p1/2.a","/gpu-data/athdom/p5/2.a", "/gpu-data/athdom/p11/2.a", "/gpu-data/athdom/p11/3.a", "/gpu-data/athdom/p16/3.a", "/gpu-data/athdom/p17/2.a", "/gpu-data/athdom/p17/3.a", "/gpu-data/athdom/p18/2.a", "/gpu-data/athdom/p18/3.a"]):
        self.cnn = cnn
        self.grid = grid
        self.batch_size = batch_size
        self.train_data = []
        #Train set
        for path in data_paths[:-2]:
            os.chdir(path)
            valid = open("valid.txt", "r")
            laser = np.genfromtxt("laserpoints.csv", delimiter = ",")
            centers = np.genfromtxt("centers.csv", delimiter = ",")
            for line in valid:
                start, end = line.strip().split(" ")
                i = int(start)
                end = int(end)
                subvideo = []
                while i <= end:
                    subvideo.append((laser[i], centers[i]))
                    i += 1
                self.train_data.append(subvideo)

        #Val set
        self.val_data = []
        os.chdir(data_paths[-2])
        valid = open("valid.txt", "r")
        laser = np.genfromtxt("laserpoints.csv", delimiter = ",")
        centers = np.genfromtxt("centers.csv", delimiter = ",")
        for line in valid:
            start, end = line.strip().split(" ")
            i = int(start)
            end = int(end)
            subvideo = []
            while i <= end:
                subvideo.append((laser[i], centers[i]))
                i += 1
            self.val_data.append(subvideo)

        #Test set
        self.test_data = []
        os.chdir(data_paths[-1])
        valid = open("valid.txt", "r")
        laser = np.genfromtxt("laserpoints.csv", delimiter = ",")
        centers = np.genfromtxt("centers.csv", delimiter = ",")
        for line in valid:
            start, end = line.strip().split(" ")
            i = int(start)
            end = int(end)
            subvideo = []
            while i <= end:
                subvideo.append((laser[i], centers[i]))
                i += 1
            self.test_data.append(subvideo)

        self.i = 0
        self.j = 0
        self.phase = 0

    def load(self, set):
        #Set is 0 for test set,
        if set == 0:
            data = self.train_data
        elif set == 1:
            data = self.val_data
            self.phase = 7
        else:
            data = self.test_data
            self.phase = 7
        flag = 0
        batchd = []
        batchl = []
        for i in range(self.batch_size):
            img = torch.zeros((img_side, img_side), dtype=torch.double)
            laser = data[self.i][self.j][0]
            laser_spots = laser.reshape((int(laser.shape[0] / 2), 2))
            in_box1 = np.logical_and(box[0][0] < laser_spots[:, 0], laser_spots[:, 0] < box[1][0])
            in_box2 = np.logical_and(box[0][1] < laser_spots[:, 1], laser_spots[:, 1] < box[1][1])
            in_box = np.logical_and(in_box1, in_box2)
            y = (laser_spots[in_box][:, 0] - min_width) / (max_width - min_width) * img_side
            x = img_side - (laser_spots[in_box][:, 1] - min_height) / (max_height - min_height) * img_side
            img[x.astype(int), y.astype(int)] = 1

            center = data[self.i][self.j][1]
            y1 = (center[0] - min_width) / (max_width - min_width)
            x1 = 1 - (center[1] - min_height) / (max_height - min_height)
            y2 = (center[2] - min_width) / (max_width - min_width)
            x2 = 1 - (center[3] - min_height) / (max_height - min_height)
            tag = torch.tensor([[x1, y1], [x2, y2]], dtype=torch.double)

            if self.phase == 0:
                batchd.append(img)
                batchl.append(tag * grid)
            elif self.phase == 1:
                batchd.append(mirrori(img))
                batchl.append(mirrorl(tag) * grid)
            else:
                s = shifts[self.phase - 2]
                batchd.append(shifti(img, int(s[0]), int(s[1])))
                batchl.append((tag + s / img_side) * grid)
            #If no need for init_hidden (LSTM hidden state initialization) then flag = 0, if need for init_hidden flag = 1, if last data flag = -1
            if self.i == len(data) - 1:
                self.phase += 1
                if self.phase > 7:
                    flag = -1
                    self.phase = 0
                else:
                    flag = 1
                self.i = 0
                self.j = 0
                return flag, torch.stack(batchd), torch.stack(batchl)
            if self.j == len(data[self.i]) - 1:
                self.j = 0
                self.i += 1
                return 1, torch.stack(batchd), torch.stack(batchl)
            self.j += 1
        return 0, torch.stack(batchd), torch.stack(batchl)
