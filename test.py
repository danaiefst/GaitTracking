import torch
import numpy as np
from matplotlib import pyplot as plt
import data_handler
from scipy.ndimage import rotate
import tracking_nn

grid = 7

def print_data(img, labels1, labels2):
    image = img.detach().clone()
    x1, y1, x2, y2 = data_handler.find_center(labels1)
    image[x1, y1] = 0.3
    image[x2, y2] = 0.3
    x1, y1, x2, y2 = data_handler.find_center(labels2)
    image[x1, y1] = 0.6
    image[x2, y2] = 0.6
    plt.imshow(image)
    plt.show(block=False)
    plt.pause(0.1)
    plt.clf()

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
    ret = 0
    for i in range(out.shape[0]):
        yh = out[i]
        p1_h = yh[0, :, :]
        p2_h = yh[3, :, :]
        detect_cell1 = p1_h.reshape(-1).argmax(axis = 0)
        detect_cell2 = p2_h.reshape(-1).argmax(axis = 0)
        x1, y1 = detect_cell1 // grid, detect_cell1 % grid
        x2, y2 = detect_cell2 // grid, detect_cell2 % grid
        ret += (x1 + out[i, 1, x1, y1] - labels[i, 0, 0] - labels[i, 0, 2]) ** 2 + (y1 + out[i, 2, x1, y1] - labels[i, 0, 1] - labels[i, 0, 3]) ** 2 + (x2 + out[i, 4, x2, y2] - labels[i, 1, 0] - labels[i, 1, 2]) ** 2 + (y2 + out[i, 5, x2, y2] - labels[i, 1, 1] - labels[i, 1, 3]) ** 2
    return ret / out.shape[0] / 2


data_paths=["/home/danai/Desktop/GaitTracking/p18/3.a"]#["/home/danai/Desktop/GaitTracking/p1/2.a","/home/danai/Desktop/GaitTracking/p5/2.a", "/home/danai/Desktop/GaitTracking/p11/2.a", "/home/danai/Desktop/GaitTracking/p11/3.a", "/home/danai/Desktop/GaitTracking/p16/3.a", "/home/danai/Desktop/GaitTracking/p17/3.a", "/home/danai/Desktop/GaitTracking/p18/2.a", "/home/danai/Desktop/GaitTracking/p18/3.a"]
data = data_handler.LegDataLoader(data_paths)
print("Loading dataset...")
test_set_x, test_set_y, val_set_x, val_set_y, _, _ = data.load(32)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
net = tracking_nn.Net(device)
net.load_state_dict(torch.load("/home/danai/Desktop/GaitTracking/model.pt", map_location=device))
dist = 0
for i in range(len(test_set_x)):
    with torch.no_grad():
        net.init_hidden(1)
        batch = test_set_x[i]
        print("Calculating validation batch", i)
        out = net(batch)
        dist += eucl_dist(out, test_set_y[i])
        #check_out(batch, out, val_set_y[i])
print("Mean dist:", dist / len(test_set_x))
