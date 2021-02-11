import data_handler, torch, os
from matplotlib import pyplot as plt
import tracking_nn

data_paths=["/home/danai/Desktop/GaitTracking/p17/2.a"]#,"/home/shit/Desktop/GaitTracking/p5/2.a", "/home/shit/Desktop/GaitTracking/p11/2.a", "/home/shit/Desktop/GaitTracking/p11/3.a", "/home/shit/Desktop/GaitTracking/p16/3.a", "/home/shit/Desktop/GaitTracking/p17/3.a", "/home/shit/Desktop/GaitTracking/p18/2.a", "/home/shit/Desktop/GaitTracking/p18/3.a"]

img_side = 112
max_height = 1.2
min_height = 0.2
max_width = 0.5
min_width = -0.5

def print_data(img, real):
    y,x = torch.where(img)
    y = (img_side - y) / img_side * (max_height - min_height) + min_height
    x = x / img_side * (max_width - min_width) + min_width
    y1,x1,y2,x2 = data_handler.find_center(real)
    y1 = (img_side - y1) / img_side * (max_height - min_height) + min_height
    y2 = (img_side - y2) / img_side * (max_height - min_height) + min_height
    x1 = x1 / img_side * (max_width - min_width) + min_width
    x2 = x2 / img_side * (max_width - min_width) + min_width
    plt.xlim(min_width, max_width)
    plt.ylim(min_height, max_height)
    plt.scatter(x,y, c = 'b', marker = '.')
    plt.scatter(x1, y1, c = 'r', marker = 'v')
    plt.scatter(x2, y2, c = 'y', marker = 'v')
    plt.show(block=False)
    plt.pause(0.2)
    plt.clf()


data_path = "/home/danai/Desktop/GaitTracking/data/"
paths = ["p17/2.a", "p18/2.a", "p18/3.a"]#["p1/2.a", "p11/2.a", "p17/2.a", "p17/3.a", "p18/2.a", "p18/3.a"]
data = data_handler.LegDataLoader(gait = 1, batch_size = 1, data_path = data_path, paths = paths)
f, input, label, states = data.load(0)
print_data(input[0], label[0])
print(states[0])
while(True):
    if f == -1:
        break
    f, input, label, states = data.load(0)
    print_data(input[0], label[0])
    print(states[0])
plt.close('all')
