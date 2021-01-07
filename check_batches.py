import data_handler, torch, os
from matplotlib import pyplot as plt
import tracking_nn

data_paths=["/home/danai/Desktop/GaitTracking/p17/3.a"]#,"/home/shit/Desktop/GaitTracking/p5/2.a", "/home/shit/Desktop/GaitTracking/p11/2.a", "/home/shit/Desktop/GaitTracking/p11/3.a", "/home/shit/Desktop/GaitTracking/p16/3.a", "/home/shit/Desktop/GaitTracking/p17/3.a", "/home/shit/Desktop/GaitTracking/p18/2.a", "/home/shit/Desktop/GaitTracking/p18/3.a"]

grid = 7
img_side = 112

def find_center(labels):
    x1 = img_side / grid * labels[0][0] + img_side / grid * labels[0][2]
    y1 = img_side / grid * labels[0][1] + img_side / grid * labels[0][3]
    x2 = img_side / grid * labels[1][0] + img_side / grid * labels[1][2]
    y2 = img_side / grid * labels[1][1] + img_side / grid * labels[1][3]
    return int(x1), int(y1), int(x2), int(y2)

def print_data(img, labels, fast=0):
    image = img.detach().clone()
    c = image.max() + 1
    print(labels)
    x1, y1, x2, y2 = find_center(labels)
    print(x1, y1, x2, y2)
    image[x1, y1] = c
    image[x2, y2] = c
    plt.imshow(image)
    if fast:
        plt.show(block=False)
        plt.pause(fast)
        plt.clf()
    else:
        plt.show()

for i in range(len(data_paths)):
    print(data_paths[i])
    data = sorted(os.listdir(data_paths[i] + "/data_cnn"), key = lambda a: int(a.split(".")[0]))
    for j in range(len(data)):
        img = torch.load(os.path.join(data_paths[i], "data_cnn", data[j]))
        labels = torch.load(os.path.join(data_paths[i], "labels_cnn", data[j]))
        plt.title(data[j])
        print_data(img, labels, 0.0001)


plt.close('all')
