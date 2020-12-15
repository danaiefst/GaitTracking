import data_handler, torch, os
from matplotlib import pyplot as plt

data_paths=["/home/shit/Desktop/GaitTracking/p1/2.a","/home/shit/Desktop/GaitTracking/p5/2.a", "/home/shit/Desktop/GaitTracking/p11/2.a", "/home/shit/Desktop/GaitTracking/p11/3.a", "/home/shit/Desktop/GaitTracking/p16/3.a", "/home/shit/Desktop/GaitTracking/p17/3.a", "/home/shit/Desktop/GaitTracking/p18/2.a", "/home/shit/Desktop/GaitTracking/p18/3.a"]

for i in range(len(data_paths)):
    os.listdir
    data = sorted(os.listdir(data_paths[i] + "/data"), key = lambda a: int(a.split(".")[0]))
    img = torch.load(os.path.join(data_paths[i], "data", data[0]))
    labels = torch.load(os.path.join(data_paths[i], "labels", data[0]))
    print(labels)
    data_handler.print_data(img, labels)
    

plt.close('all')
