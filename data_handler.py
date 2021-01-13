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

def transformi(img):
    ret = []
    for s in shifts:
        #print(int(s[0]), s[1])
        new_img = shifti(img, int(s[0]), int(s[1]))
        ret.append(new_img)
    ret.append(mirrori(img))
    return ret

def transforml(label):
    ret = []
    for s in shifts:
        new_label = label + s / img_side
        if new_label[0, 0] <= 1 and new_label[0, 1] <= 1 and new_label[1][0] <= 1 and new_label[1][1] <= 1:
            ret.append(new_label)
        else:
            print(label, label + s / img_side)
            print("Out of bounds", s)
    ret.append(mirrorl(label))
    return ret

def find_center(label):
    ret = label / grid * img_side
    return ret[0], ret[1], ret[2], ret[3]

class LegDataLoader():

    """expecting to find at data_paths a data and a labels folder"""
    def __init__(self, grid = 7, cnn = 0, online = 0, data_paths=["/gpu-data/athdom/p1/2.a","/gpu-data/athdom/p5/2.a", "/gpu-data/athdom/p11/2.a", "/gpu-data/athdom/p11/3.a", "/gpu-data/athdom/p16/3.a", "/gpu-data/athdom/p17/2.a", "/gpu-data/athdom/p17/3.a", "/gpu-data/athdom/p18/2.a", "/gpu-data/athdom/p18/3.a"]):
        self.data_paths = data_paths
        self.cnn = cnn
        self.grid = grid
        if online:
            os.chdir(data_paths[0])
            valid = open("valid.txt", "r")
            laser = np.genfromtxt("laserpoints.csv", delimiter = ",")
            centers = np.genfromtxt("centers.csv", delimiter = ",")
            self.online_data = []
            for line in valid:
                start, end = line.strip().split(" ")
                i = int(start)
                end = int(end)
                subvideo = []
                while i <= end:
                    subvideo.append((laser[i], centers[i]))
                    i += 1
                self.online_data.append(subvideo)
            self.online_i = 0
            self.online_j = 0


    def make_batches(self, batch_size, vid_i, video):
        vid_batchd = []
        vid_batchl = []
        i = batch_size
        batch_data = []
        batch_labels = []
        print("Loading video", vid_i, "...")
        prev_frame = None
        for frame in video:
            frame_i = int(frame.split(".")[0])
            #print(i, frame_i)
            if i == 0 or prev_frame and prev_frame + 1 != frame_i:
                #print(i, prev_frame, frame_i)
                vid_batchd.append(torch.stack(batch_data, dim = 0))
                vid_batchl.append(torch.stack(batch_labels, dim = 0))
                if self.cnn:
                    batch_data = [torch.load(self.data_paths[vid_i] + "/data_cnn/" + frame)]
                    batch_labels = [grid * torch.load(self.data_paths[vid_i] + "/labels_cnn/" + frame)]
                else:
                    batch_data = [torch.load(self.data_paths[vid_i] + "/data/" + frame)]
                    batch_labels = [grid * torch.load(self.data_paths[vid_i] + "/labels/" + frame)]
                i = batch_size - 1
            else:
                if self.cnn:
                    batch_data.append(torch.load(self.data_paths[vid_i] + "/data_cnn/" + frame))
                    batch_labels.append(grid * torch.load(self.data_paths[vid_i] + "/labels_cnn/" + frame))
                else:
                    batch_data.append(torch.load(self.data_paths[vid_i] + "/data/" + frame))
                    batch_labels.append(grid * torch.load(self.data_paths[vid_i] + "/labels/" + frame))
                i -= 1
            prev_frame = frame_i
        if batch_data != []:
            vid_batchd.append(torch.stack(batch_data, dim = 0).to(self.device))
            vid_batchl.append(torch.stack(batch_labels, dim = 0).to(self.device))
        return vid_batchd, vid_batchl

    def load(self, batch_size):
        self.data = []

        for path in self.data_paths:
            if self.cnn:
                self.data.append(sorted(os.listdir(path + "/data_cnn"), key = lambda a: int(a.split(".")[0])))
            else:
                self.data.append(sorted(os.listdir(path + "/data"), key = lambda a: int(a.split(".")[0])))
        train_set_x = []
        train_set_y = []

        for vid_i, video in enumerate(self.data[:-2]):
            vid_batchd, vid_batchl = self.make_batches(batch_size, vid_i, video)
            train_set_x.extend(vid_batchd)
            train_set_y.extend(vid_batchl)

        vid_batchd, vid_batchl = self.make_batches(batch_size, len(self.data) - 2, self.data[-2])
        val_set_x = vid_batchd
        val_set_y = vid_batchl
        vid_batchd, vid_batchl = self.make_batches(batch_size, len(self.data) - 1, self.data[-1])
        test_set_x = vid_batchd
        test_set_y = vid_batchl
        return train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, test_set_y

    def load_online(self):
        img = torch.zeros((img_side, img_side), dtype=torch.double)
        laser = self.online_data[self.online_i][self.online_j][0]
        laser_spots = laser.reshape((int(laser.shape[0] / 2), 2))
        in_box1 = np.logical_and(box[0][0] < laser_spots[:, 0], laser_spots[:, 0] < box[1][0])
        in_box2 = np.logical_and(box[0][1] < laser_spots[:, 1], laser_spots[:, 1] < box[1][1])
        in_box = np.logical_and(in_box1, in_box2)
        y = (laser_spots[in_box][:, 0] - min_width) / (max_width - min_width) * img_side
        x = img_side - (laser_spots[in_box][:, 1] - min_height) / (max_height - min_height) * img_side
        img[x.astype(int), y.astype(int)] = 1
        img = img.view(1, *img.shape)

        center = self.online_data[self.online_i][self.online_j][1]
        y1 = (center[0] - min_width) / (max_width - min_width)
        x1 = 1 - (center[1] - min_height) / (max_height - min_height)
        y2 = (center[2] - min_width) / (max_width - min_width)
        x2 = 1 - (center[3] - min_height) / (max_height - min_height)
        tag = torch.tensor([[x1, y1], [x2, y2]], dtype=torch.double)
        tag = tag.view(1, *tag.shape)

        #If no need for init_hidden (LSTM hidden state initialization) then flag = 0, if need for init_hidden flag = 1, if last data flag = -1
        if self.online_i == len(self.online_data) - 1:
            return -1, img, grid * tag
        if self.online_j == len(self.online_data[self.online_i]) - 1:
            self.online_j = 0
            self.online_i += 1
            return 1, img, grid * tag
        self.online_j += 1
        return 0, img, grid * tag
