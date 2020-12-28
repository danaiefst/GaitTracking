import torch
import numpy as np
from matplotlib import pyplot as plt
import data_handler
from scipy.ndimage import rotate
import tracking_nn
import time

grid = 7

def print_data(img, found, real):
    y,x = torch.where(img)
    y = (112 - y) / 112.0 * 1.1 + 0.1
    x = x / 112.0 - 0.5
    y1,x1,y2,x2 = data_handler.find_center(real)
    y1 = (112 - y1) / 112.0 * 1.1 + 0.1
    y2 = (112 - y2) / 112.0 * 1.1 + 0.1
    x1 = x1 / 112 - 0.5
    x2 = x2 / 112 - 0.5
    y1h,x1h,y2h,x2h = data_handler.find_center(found)
    y1h = (112 - y1h) / 112.0 * 1.1 + 0.1
    y2h = (112 - y2h) / 112.0 * 1.1 + 0.1
    x1h = x1h / 112.0 - 0.5
    x2h = x2h / 112.0 - 0.5
    plt.xlim(-0.5, 0.5)
    plt.ylim(0.1, 1.2)
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
        y = batch[i]
        yh = out[i]
        p1_h = yh[0, :, :]
        p2_h = yh[3, :, :]
        detect_cell1 = p1_h.reshape(-1).argmax(axis = 0)
        detect_cell2 = p2_h.reshape(-1).argmax(axis = 0)
        x_cell1, y_cell1 = detect_cell1 // grid, detect_cell1 % grid
        x_cell2, y_cell2 = detect_cell2 // grid, detect_cell2 % grid
        print_data(y, [[x_cell1, y_cell1, out[i][1, x_cell1, y_cell1], out[i][2, x_cell1, y_cell1]], [x_cell2, y_cell2, out[i][4, x_cell2, y_cell2], out[i][5, x_cell2, y_cell2]]], label[i])

def eucl_dist(out, labels):
    ret1 = 0
    ret2 = 0
    ret = []
    for i in range(out.shape[0]):
        yh = out[i]
        p1_h = yh[0, :, :]
        p2_h = yh[3, :, :]
        detect_cell1 = p1_h.reshape(-1).argmax(axis = 0)
        detect_cell2 = p2_h.reshape(-1).argmax(axis = 0)
        x1, y1 = detect_cell1 // grid, detect_cell1 % grid
        x2, y2 = detect_cell2 // grid, detect_cell2 % grid
        ret1 += torch.sqrt((x1 + out[i, 1, x1, y1] - labels[i, 0, 0] - labels[i, 0, 2]) ** 2 + (y1 + out[i, 2, x1, y1] - labels[i, 0, 1] - labels[i, 0, 3]) ** 2).item()
        ret2 += torch.sqrt((x2 + out[i, 4, x2, y2] - labels[i, 1, 0] - labels[i, 1, 2]) ** 2 + (y2 + out[i, 5, x2, y2] - labels[i, 1, 1] - labels[i, 1, 3]) ** 2).item()
        ret.append(ret1 / out.shape[0])
        ret.append(ret2 / out.shape[0])
    return ret

def median(l):
    # l sorted list
    if len(l) % 2 == 0:
        return (l[len(l) // 2] + l[len(l) // 2 - 1]) / 2
    else:
        return l[len(l) // 2]


data_paths=["/home/danai/Desktop/GaitTracking/p1/2.a"]#,"/home/shit/Desktop/GaitTracking/p5/2.a", "/home/shit/Desktop/GaitTracking/p11/2.a", "/home/shit/Desktop/GaitTracking/p11/3.a", "/home/shit/Desktop/GaitTracking/p16/3.a", "/home/shit/Desktop/GaitTracking/p17/3.a", "/home/shit/Desktop/GaitTracking/p18/2.a", "/home/shit/Desktop/GaitTracking/p18/3.a"]
data = data_handler.LegDataLoader(data_paths = data_paths, cnn=1)
print("Loading dataset...")
tx, ty, vx, vy, _, _ = data.load(32)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net = tracking_nn.CNN(device)
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
        print("Time taken:", time.time() - t)
        all_dists.extend(eucl_dist(out, vy[i].to(device)))
        check_out(batch.to(torch.device("cpu")), out.to(torch.device("cpu")), vy[i].to(torch.device("cpu")))

all_dists.sort()
print("Mean dist:", sum(all_dists) / len(all_dists) / 7, "Max dist:", max(all_dists) / 7, "Median dist:", median(all_dists) / 7)
