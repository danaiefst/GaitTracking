import data_handler, torch, os
from matplotlib import pyplot as plt
import tracking_nn

data_paths=["/home/danai/Desktop/GaitTracking/p1/2.a"]#,"/home/shit/Desktop/GaitTracking/p5/2.a", "/home/shit/Desktop/GaitTracking/p11/2.a", "/home/shit/Desktop/GaitTracking/p11/3.a", "/home/shit/Desktop/GaitTracking/p16/3.a", "/home/shit/Desktop/GaitTracking/p17/3.a", "/home/shit/Desktop/GaitTracking/p18/2.a", "/home/shit/Desktop/GaitTracking/p18/3.a"]

"""for i in range(len(data_paths)):
    print(data_paths[i])
    data = sorted(os.listdir(data_paths[i] + "/data_cnn"), key = lambda a: int(a.split(".")[0]))
    for j in range(1967, len(data)):
        img = torch.load(os.path.join(data_paths[i], "data_cnn", data[j]))
        labels = torch.load(os.path.join(data_paths[i], "labels_cnn", data[j]))



plt.close('all')"""

data = data_handler.LegDataLoader(data_paths = data_paths, cnn=1)
tx, ty, vx, vy, _, _ = data.load(32)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net = tracking_nn.Net(device)

x = torch.zeros(1, 6, 7, 7)

img = ty[0][0]
print(img.view(1, img.shape[0], img.shape[1])[:, 0, 0])

x[0, 0, img[0][0].long(), img[0][1].long()] = 1
x[0, 3, img[1][0].long(), img[1][1].long()] = 1
x[0, 3, 0, 0] = 0.8
x[0, 1, img[0][0].long(), img[0][1].long()] = img[0][2]
x[0, 2, img[0][0].long(), img[0][1].long()] = img[0][3]
x[0, 4, img[1][0].long(), img[1][1].long()] = img[1][2]
x[0, 5, img[1][0].long(), img[1][1].long()] = img[1][3]

data_handler.print_data(tx[0][0], ty[0][0])
data_handler.print_data(tx[0][10], ty[0][10])
print(net.loss(x, ty[0][10].view(1, img.shape[0], img.shape[1])))
