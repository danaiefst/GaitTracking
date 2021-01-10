import torch
import numpy as np
from matplotlib import pyplot as plt
import data_handler
from scipy.ndimage import rotate
import tracking_nn
import time

img_side = (img_side - 1)
max_height = 1.2
min_height = 0.1
max_width = 0.5
min_width = -0.5

def print_data(img, found, real):
    y,x = torch.where(img)
    y = (img_side - 1 - y) / (img_side - 1) * (max_height - min_height) + min_height
    x = x / (img_side - 1.0) - max_width
    y1,x1,y2,x2 = real[0, 1], real[0, 0], real[1, 1], real[1, 0]
    y1 = ((img_side - 1) - y1) / (img_side - 1.0) * (max_height - min_height) + min_height
    y2 = ((img_side - 1) - y2) / (img_side - 1.0) * (max_height - min_height) + min_height
    x1 = x1 / (img_side - 1) - max_width
    x2 = x2 / (img_side - 1) - max_width
    y1h,x1h,y2h,x2h = found[1], found[0], found[3], found[2]
    y1h = ((img_side - 1) - y1h) / (img_side - 1.0) * (max_height - min_height) + min_height
    y2h = ((img_side - 1) - y2h) / (img_side - 1.0) * (max_height - min_height) + min_height
    x1h = x1h / (img_side - 1.0) - max_width
    x2h = x2h / (img_side - 1.0) - max_width
    plt.xlim(min_width, max_width)
    plt.ylim(min_height, max_height)
    plt.scatter(x,y, c = 'b', marker = '.')
    plt.scatter(x1, y1, c = 'r', marker = 'v')
    plt.scatter(x2, y2, c = 'y', marker = 'v')
    plt.scatter(x1h, y1h, c = 'r', marker = 'o')
    plt.scatter(x2h, y2h, c = 'y', marker = 'o')
    plt.show()
    #plt.show(block=False)
    #plt.pause(0.1)
    #plt.clf()

def check_out(batch, out, label):
    for i in range(out.shape[0]):
        print_data(y, out[i], label[i])

def eucl_dist(out, labels):
    return (torch.sqrt(((out[:, :2] - labels[:, 0]) ** 2).sum(axis = 1)).sum() + torch.sqrt(((out[:, 2:] - labels[:, 1]) ** 2).sum(axis = 1)).sum()) / out.shape[0] / 2

def median(l):
    # l sorted list
    if len(l) % 2 == 0:
        return (l[len(l) // 2] + l[len(l) // 2 - 1]) / 2
    else:
        return l[len(l) // 2]


data_paths=["/home/danai/Desktop/GaitTracking/p18/3.a"]#,"/home/shit/Desktop/GaitTracking/p5/2.a", "/home/shit/Desktop/GaitTracking/p11/2.a", "/home/shit/Desktop/GaitTracking/p11/3.a", "/home/shit/Desktop/GaitTracking/p16/3.a", "/home/shit/Desktop/GaitTracking/p17/3.a", "/home/shit/Desktop/GaitTracking/p18/2.a", "/home/shit/Desktop/GaitTracking/p18/3.a"]
data = data_handler.LegDataLoader(data_paths = data_paths)
print("Loading dataset...")
tx, ty, vx, vy, _, _ = data.load(32)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net = tracking_nn.Net(device)
#net.load_state_dict(torch.load("/home/shit/Desktop/GaitTracking/model.pt"))
#net.to(device)
net.load_state_dict(torch.load("/home/danai/Desktop/GaitTracking/model.pt", map_location=device))
all_dists = []
for i in range(len(vx)):
    with torch.no_grad():
        net.init_hidden(1)
        batch = vx[i].to(device)
        print("Calculating validation batch", i)
        t = time.time()
        out = net(batch)
        #print("Time taken:", time.time() - t)
        all_dists.extend(eucl_dist(out, vy[i].to(device)))
        #check_out(batch.to(torch.device("cpu")), out.to(torch.device("cpu")), vy[i].to(torch.device("cpu")))

all_dists.sort()
print("Mean dist:", sum(all_dists) / len(all_dists) / (img_side - 1), "Max dist:", max(all_dists) / (img_side - 1), "Median dist:", median(all_dists) / (img_side - 1))
