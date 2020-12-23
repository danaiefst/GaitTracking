import torch
import os
from scipy import ndimage
from matplotlib import pyplot as plt
from math import pi, sin, cos

max_height = 1.2
min_height = 0.1
max_width = 0.5
min_width = -0.5
img_side = 112
grid = 7
shifts = [(-15, 10), (-10, -10), (-10, 10), (-5, 10), (-7, 5), (2, 10)]
torch.set_default_dtype(torch.double)


def find_center(labels):
    x1 = img_side / grid * labels[0][0] + img_side / grid * labels[0][2]
    y1 = img_side / grid * labels[0][1] + img_side / grid * labels[0][3]
    x2 = img_side / grid * labels[1][0] + img_side / grid * labels[1][2]
    y2 = img_side / grid * labels[1][1] + img_side / grid * labels[1][3]
    return int(x1), int(y1), int(x2), int(y2)

def mirrori(img):
    new_img = img.flip(-1)
    return new_img

def mirrorl(labels):
    new_labels = torch.tensor([[labels[1][0], grid - 1 - labels[1][1], labels[1][2], 1 - labels[1][3]], [labels[0][0], grid - 1 - labels[0][1], labels[0][2], 1 - labels[0][3]]], dtype=torch.double)
    return new_labels

def shift_coord(xcell, xcenter, xshift):
    x = img_side / grid * xcell + img_side / grid * xcenter + xshift
    temp = x / img_side * grid
    return int(temp), temp % 1

def rot_coord(x, y, angle):
    a = angle * pi / 180
    x -= 99
    y -= 50
    new_x = x * cos(a) - y * sin(a) + 99
    new_y = x * sin(a) + y * cos(a) + 49
    return new_x, new_y


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

def shiftl(labels, x, y):
    xcell1, xcenter1 = shift_coord(labels[0][0], labels[0][2], x)
    xcell2, xcenter2 = shift_coord(labels[1][0], labels[1][2], x)
    ycell1, ycenter1 = shift_coord(labels[0][1], labels[0][3], y)
    ycell2, ycenter2 = shift_coord(labels[1][1], labels[1][3], y)
    new_labels = torch.tensor([[xcell1, ycell1, xcenter1, ycenter1], [xcell2, ycell2, xcenter2, ycenter2]], dtype=torch.double)
    return new_labels

def rotatei(img, angle):
    # Rotate img
    new_img = torch.zeros(img.shape, dtype=torch.double)
    x, y = torch.where(img == 1)
    new_x, new_y = rot_coord(x, y, angle)
    new_img[new_x.long(), new_y.long()] = 1
    return new_img

def rotatel(labels, angle):
    # Rotate labels
    x1 = img_side / grid * labels[0][0] + img_side / grid * labels[0][2]
    y1 = img_side / grid * labels[0][1] + img_side / grid * labels[0][3]
    x2 = img_side / grid * labels[1][0] + img_side / grid * labels[1][2]
    y2 = img_side / grid * labels[1][1] + img_side / grid * labels[1][3]
    x1, y1 = rot_coord(x1, y1, angle)
    x2, y2 = rot_coord(x2, y2, angle)
    temp = x1 / img_side * grid
    xcell1, xcenter1 = int(temp), temp % 1
    temp = y1 / img_side * grid
    ycell1, ycenter1 = int(temp), temp % 1
    temp = x2 / img_side * grid
    xcell2, xcenter2 = int(temp), temp % 1
    temp = y2 / img_side * grid
    ycell2, ycenter2 = int(temp), temp % 1
    new_labels = torch.tensor([[xcell1, ycell1, xcenter1, ycenter1], [xcell2, ycell2, xcenter2, ycenter2]], dtype=torch.double)
    return new_labels

def print_data(img, labels):
    image = img.detach().clone()
    x1, y1, x2, y2 = find_center(labels)
    image[x1, y1] = 0.2
    image[x2, y2] = 0.8
    plt.imshow(image)
    #plt.show()
    plt.show(block=False)
    plt.pause(0.5)
    plt.clf()

def transformi(img):
    ret = []
    for s in shifts:
        new_img = shifti(img, *s)
        ret.append(new_img)
    ret.append(mirrori(img))
    return ret

def transforml(label):
    ret = []
    for s in shifts:
        new_label = shiftl(label, *s)
        if new_label[0][0] <= 6 and new_label[0][1] <= 6 and new_label[1][0] <= 6 and new_label[1][1] <= 6:
            ret.append(new_label)
        else:
            print("Out of bounds", s)
    ret.append(mirrorl(label))
    return ret

class LegDataLoader():

    """expecting to find at data_paths a data and a labels folder"""
    def __init__(self, data_paths=["/gpu-data/athdom/p1/2.a","/gpu-data/athdom/p5/2.a", "/gpu-data/athdom/p11/2.a", "/gpu-data/athdom/p11/3.a", "/gpu-data/athdom/p16/3.a", "/gpu-data/athdom/p17/3.a", "/gpu-data/athdom/p18/2.a", "/gpu-data/athdom/p18/3.a"]):
        self.data_paths = data_paths

    def load(self, batch_size = 64):
        self.data = []
        for path in self.data_paths:
            self.data.append(sorted(os.listdir(path + "/data"), key = lambda a: int(a.split(".")[0])))
        train_set_x = []
        train_set_y = []
        val_set_x = []
        val_set_y = []
        test_set_x = []
        test_set_y = []
        batch_size = batch_size
        for vid_i, video in enumerate(self.data):
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
                    batch_data = [torch.load(self.data_paths[vid_i] + "/data/" + frame)]
                    batch_labels = [torch.load(self.data_paths[vid_i] + "/labels/" + frame)]
                    i = batch_size - 1
                else:
                    batch_data.append(torch.load(self.data_paths[vid_i] + "/data/" + frame))
                    batch_labels.append(torch.load(self.data_paths[vid_i] + "/labels/" + frame))
                    i -= 1
                prev_frame = frame_i
            if batch_data != []:
                vid_batchd.append(torch.stack(batch_data, dim = 0))
                vid_batchl.append(torch.stack(batch_labels, dim = 0))
            train_set_x.extend(vid_batchd[:int(len(vid_batchd) * 0.7)])
            train_set_y.extend(vid_batchl[:int(len(vid_batchl) * 0.7)])
            val_set_x.extend(vid_batchd[int(len(vid_batchd) * 0.7) : int(len(vid_batchd) * 0.85)])
            val_set_y.extend(vid_batchl[int(len(vid_batchl) * 0.7) : int(len(vid_batchl) * 0.85)])
            test_set_x.extend(vid_batchd[int(len(vid_batchd) * 0.85):])
            test_set_y.extend(vid_batchl[int(len(vid_batchl) * 0.85):])
        return train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, test_set_y
