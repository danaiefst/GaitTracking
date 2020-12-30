import data_handler, torch, os
from matplotlib import pyplot as plt
import tracking_nn

data_paths=["/home/danai/Desktop/GaitTracking/p18/2.a"]#,"/home/shit/Desktop/GaitTracking/p5/2.a", "/home/shit/Desktop/GaitTracking/p11/2.a", "/home/shit/Desktop/GaitTracking/p11/3.a", "/home/shit/Desktop/GaitTracking/p16/3.a", "/home/shit/Desktop/GaitTracking/p17/3.a", "/home/shit/Desktop/GaitTracking/p18/2.a", "/home/shit/Desktop/GaitTracking/p18/3.a"]

for i in range(len(data_paths)):
    print(data_paths[i])
    data = sorted(os.listdir(data_paths[i] + "/data"), key = lambda a: int(a.split(".")[0]))
    for j in range(len(data)):
        img = torch.load(os.path.join(data_paths[i], "data", data[j]))
        labels = torch.load(os.path.join(data_paths[i], "labels", data[j]))
        plt.title(data[j])
        data_handler.print_data(img, labels, 0.1)


plt.close('all')
