import torch
import numpy as np
from matplotlib import pyplot as plt
import data_handler
from scipy.ndimage import rotate
import tracking_nn

def check_out(batch, out):
    for i in range(out.shape[0]):
        y = batch[i]
        yh = out[i]
        p1_h = yh[0, :, :]
        p2_h = yh[3, :, :]
        detect_cell1 = p1_h.reshape(-1).argmax(axis = 0)
        detect_cell2 = p2_h.reshape(-1).argmax(axis = 0)
        x_cell1, y_cell1 = detect_cell1 // 8, detect_cell1 % 8
        x_cell2, y_cell2 = detect_cell2 // 8, detect_cell2 % 8
        data_handler.print_data(y, [[x_cell1, y_cell1, out[i][1, x_cell1, y_cell1], out[i][2, x_cell1, y_cell1]], [x_cell2, y_cell2, out[i][4, x_cell2, y_cell2], out[i][5, x_cell2, y_cell2]]])

data_paths=["/home/danai/Desktop/GaitTracking/p1/2.a"]#,"/home/danai/Desktop/GaitTracking/p5/2.a", "/home/danai/Desktop/GaitTracking/p11/2.a", "/home/danai/Desktop/GaitTracking/p11/3.a", "/home/danai/Desktop/GaitTracking/p16/3.a", "/home/danai/Desktop/GaitTracking/p17/3.a", "/home/danai/Desktop/GaitTracking/p18/2.a", "/home/danai/Desktop/GaitTracking/p18/3.a"]
data = data_handler.LegDataLoader(data_paths)
print("Loading dataset...")
_, _, val_set_x, _, _, _ = data.load(32)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
net = tracking_nn.Net(device)
net.load_state_dict(torch.load("/home/danai/Desktop/GaitTracking/model.pt", map_location=device))
for batch in val_set_x:
    out = net(batch)
    check_out(batch, out)














yh = torch.zeros((6, 8, 8), dtype=torch.double)
x1, y1, x2, y2 = int(label1[0][0]), int(label1[0][1]), int(label1[1][0]), int(label1[1][1])
yh[0, x1, y1] = 1
yh[3, x2, y2] = 1
yh[1, x1, y1] = label1[0][2]
yh[2, x1, y1] = label1[0][3]
yh[1, x2, y2] = label1[1][2]
yh[2, x2, y2] = label1[1][3]

def loss(y_h, y):
    y = y.to(torch.double)
    y_h = y_h.to(torch.double)
    """y are the data labels (format: [[[x_cell1, y_cell1, x_center1, y_center1], [x_cell2, y_cell2, x_center2, y_center2]], ...]), y_h (y hat) is the nn's output"""
    #print(y_h[:, 0, :, :])
    p1_h = y_h[:, 0, :, :]
    p2_h = y_h[:, 3, :, :]
    p1 = torch.zeros(p1_h.shape)
    p2 = torch.zeros(p2_h.shape)
    p1[torch.arange(y.shape[0]), y[:, 0, 0].long(), y[:, 0, 1].long()] = 1
    p2[torch.arange(y.shape[0]), y[:, 1, 0].long(), y[:, 1, 1].long()] = 1
    prob_loss = ((p1 - p1_h) ** 2).sum() + ((p2 - p2_h) ** 2).sum()
    detect_cell1 = p1_h.reshape((p1.size(0), -1)).argmax(axis = 1)
    detect_cell2 = p2_h.reshape((p2.size(0), -1)).argmax(axis = 1)
    detect_cell1 = torch.stack((detect_cell1 // 8, detect_cell1 % 8), dim = 1)
    detect_cell2 = torch.stack((detect_cell2 // 8, detect_cell2 % 8), dim = 1)
    #print(y, detect_cell1, detect_cell2)
    detect_loss = 64 * ((y_h[torch.arange(p1.size(0)), 1:3, detect_cell1[:, 0], detect_cell1[:, 1]] - y[:, 0, 2:]) ** 2).sum() + 64 * ((y_h[torch.arange(p1.size(0)), 4:, detect_cell2[:, 0], detect_cell2[:, 1]] - y[:, 1, 2:]) ** 2).sum() + ((detect_cell1.double() - y[:, 0, :2]) ** 2).sum() + ((detect_cell2.double() - y[:, 1, :2]) ** 2).sum()
    print(prob_loss / y.shape[0], detect_loss / y.shape[0])
    return prob_loss + detect_loss
