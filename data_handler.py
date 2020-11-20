import torch
import os
import random
from matplotlib import pyplot as plt

random.seed(0)

max_height = 1.2
min_height = 0.1
max_width = 0.5
min_width = -0.5
img_side = 100
grid = 8

def find_center(labels):
    x1 = img_side / grid * labels[0][0] + img_side / grid * labels[0][2]
    y1 = img_side / grid * labels[0][1] + img_side / grid * labels[0][3]
    x2 = img_side / grid * labels[1][0] + img_side / grid * labels[1][2]
    y2 = img_side / grid * labels[1][1] + img_side / grid * labels[1][3]
    return int(x1), int(y1), int(x2), int(y2)

def mirror(img, labels):
    new_img = img.flip(-1)
    new_labels = torch.tensor([[labels[1][0], grid - 1 - labels[1][1], labels[1][2], 1 - labels[1][3]], [labels[0][0], grid - 1 - labels[0][1], labels[0][2], 1 - labels[0][3]]])
    return new_img, new_labels

def shift_coord(xcell, xcenter, xshift):
    x = img_side / grid * xcell + img_side / grid * xcenter + xshift
    temp = x / img_side * grid
    return int(temp), temp % 1

def shift(img, labels, x, y):
    new_img = torch.zeros(img.shape)
    new_img[x:, y:] = img[:img.shape[0] - x, :img.shape[1] - y]
    xcell1, xcenter1 = shift_coord(labels[0][0], labels[0][2], x)
    xcell2, xcenter2 = shift_coord(labels[1][0], labels[1][2], y)
    ycell1, ycenter1 = shift_coord(labels[0][1], labels[0][3], x)
    ycell2, ycenter2 = shift_coord(labels[1][1], labels[1][3], y)
    new_labels = torch.tensor([[xcell1, ycell1, xcenter1, ycenter1], [xcell2, ycell2, xcenter2, ycenter2]])
    return new_img, new_labels

def rotate(img, labels, angle):
    pass

def print_data(img, labels):
    image = img.detach().clone()
    x1, y1, x2, y2 = find_center(labels)
    image[x1, y1] = 0.5
    image[x2, y2] = 0.5
    print(image)
    plt.imshow(image, cmap='gray')
    plt.show()


class LegDataLoader():

    """expecting to find at data_paths a data and a labels folder"""
    def __init__(self, data_paths=["/home/athdom/GaitTracking/p1/2.a","/home/athdom/GaitTracking/p5/2.a", "/home/athdom/GaitTracking/p11/2.a", "/home/athdom/GaitTracking/p11/3.a", "/home/athdom/GaitTracking/p16/3.a", "/home/athdom/GaitTracking/p17/3.a", "/home/athdom/GaitTracking/p18/2.a", "/home/athdom/GaitTracking/p18/3.a"]):
        self.data_paths = data_paths

    def load(self, batch_size = 64):
        self.data = []
        for path in self.data_paths:
            self.data.append(sorted(os.listdir(path + "/data"), key = lambda a: int(a.split(".")[0])))
        self.batched_data = []
        self.batched_labels = []
        self.batch_size = batch_size
        i = batch_size
        batch_data = []
        batch_labels = []
        for vid_i, video in enumerate(self.data):
            prev_frame = None
            for frame in video:
                frame_i = int(frame.split(".")[0])
                if i == 0 or prev_frame and prev_frame + 1 != frame_i:
                    #print(i, prev_frame, frame_i)
                    self.batched_data.append(torch.stack(batch_data, dim = 0))
                    self.batched_labels.append(torch.stack(batch_labels, dim = 0))
                    batch_data = [torch.load(self.data_paths[vid_i] + "/data/" + frame)]
                    batch_labels = [torch.load(self.data_paths[vid_i] + "/labels/" + frame)]
                    i = batch_size - 1
                else:
                    batch_data.append(torch.load(self.data_paths[vid_i] + "/data/" + frame))
                    batch_labels.append(torch.load(self.data_paths[vid_i] + "/labels/" + frame))
                    i -= 1
                prev_frame = frame_i

        s = list(zip(self.batched_data, self.batched_labels))
        random.shuffle(s)
        self.batched_data, self.batched_labels = zip(*s)
        return self.batched_data[:int(len(self.batched_data) * 0.7)], self.batched_labels[:int(len(self.batched_labels) * 0.7)], self.batched_data[int(len(self.batched_data) * 0.7):int(len(self.batched_data) * 0.85)], self.batched_labels[int(len(self.batched_labels) * 0.7):int(len(self.batched_labels) * 0.85)], self.batched_data[int(len(self.batched_data) * 0.85):], self.batched_labels[int(len(self.batched_labels) * 0.85):]
