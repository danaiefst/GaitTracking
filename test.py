import torch
from matplotlib import pyplot as plt
import data_handler

img = torch.load("/home/danai/Desktop/GaitTracking/p5/2.a/data/2185.pt")
label = torch.load("/home/danai/Desktop/GaitTracking/p5/2.a/labels/2185.pt")

data_handler.print_data(img, label)
y, yl = data_handler.mirror(img, label)
data_handler.print_data(y, yl)
