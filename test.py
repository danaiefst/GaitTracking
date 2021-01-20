import torch
from matplotlib import pyplot as plt
import data_handler
import tracking_nn
import time

img_side = 112
max_height = 1.2
min_height = 0.2
max_width = 0.5
min_width = -0.5
grid = 7

def print_data(img, found, real):
    y,x = torch.where(img)
    y = (img_side - y) / img_side * (max_height - min_height) + min_height
    x = x / img_side * (max_width - min_width) + min_width
    y1,x1,y2,x2 = data_handler.find_center(real)
    y1 = (img_side - y1) / img_side * (max_height - min_height) + min_height
    y2 = (img_side - y2) / img_side * (max_height - min_height) + min_height
    x1 = x1 / img_side * (max_width - min_width) + min_width
    x2 = x2 / img_side * (max_width - min_width) + min_width
    y1h,x1h,y2h,x2h = data_handler.find_center(found)
    y1h = (img_side - y1h) / img_side * (max_height - min_height) + min_height
    y2h = (img_side - y2h) / img_side * (max_height - min_height) + min_height
    x1h = x1h / img_side * (max_width - min_width) + min_width
    x2h = x2h / img_side * (max_width - min_width) + min_width
    plt.xlim(min_width, max_width)
    plt.ylim(min_height, max_height)
    plt.scatter(x,y, c = 'b', marker = '.')
    plt.scatter(x1, y1, c = 'r', marker = 'v')
    plt.scatter(x2, y2, c = 'y', marker = 'v')
    plt.scatter(x1h, y1h, c = 'r', marker = 'o')
    plt.scatter(x2h, y2h, c = 'y', marker = 'o')
    plt.show()
    #plt.show(block=False)
    #plt.pause(0.01)
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
        print_data(y, [[x_cell1 + out[i][1, x_cell1, y_cell1], y_cell1 + out[i][2, x_cell1, y_cell1]], [x_cell2 + out[i][4, x_cell2, y_cell2], y_cell2 + out[i][5, x_cell2, y_cell2]]], label[i])


def eucl_dist(out, labels):
    ret = []
    for i in range(out.shape[0]):
        yh = out[i]
        p1_h = yh[0, :, :]
        p2_h = yh[3, :, :]
        detect_cell1 = p1_h.reshape(-1).argmax(axis = 0)
        detect_cell2 = p2_h.reshape(-1).argmax(axis = 0)
        x1, y1 = detect_cell1 // grid, detect_cell1 % grid
        x2, y2 = detect_cell2 // grid, detect_cell2 % grid
        ret.append(torch.sqrt((x1 + out[i, 1, x1, y1] - labels[i, 0, 0]) ** 2 + (y1 + out[i, 2, x1, y1] - labels[i, 0, 1]) ** 2).item())
        ret.append(torch.sqrt((x2 + out[i, 4, x2, y2] - labels[i, 1, 0]) ** 2 + (y2 + out[i, 5, x2, y2] - labels[i, 1, 1]) ** 2).item())
    return ret

def median(l):
    # l sorted list
    if len(l) % 2 == 0:
        return (l[len(l) // 2] + l[len(l) // 2 - 1]) / 2
    else:
        return l[len(l) // 2]


data_paths=["/home/danai/Desktop/GaitTracking/p1/2.a", "/home/danai/Desktop/GaitTracking/p18/2.a", "/home/danai/Desktop/GaitTracking/p18/3.a"]
data = data_handler.LegDataLoader(data_paths = data_paths)
print("Loading dataset...")
tx, ty, vx, vy, tsx, tsy = data.load(32)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cnn = tracking_nn.CNN()
rnn = tracking_nn.RNN()
net = tracking_nn.Net(device, cnn, rnn)
#net.load_state_dict(torch.load("/home/shit/Desktop/GaitTracking/model.pt"))
#net.to(device)
net.load_state_dict(torch.load("/home/danai/Desktop/GaitTracking/best_model.pt", map_location=device)).to(device)
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
print("Mean dist:", sum(all_dists) / len(all_dists) / grid, "Max dist:", max(all_dists) / grid, "Median dist:", median(all_dists) / grid)
