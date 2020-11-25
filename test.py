import torch
import numpy as np
from matplotlib import pyplot as plt
import data_handler
from scipy.ndimage import rotate

img = torch.load("/home/danai/Desktop/GaitTracking/p5/2.a/data/2185.pt")
label = torch.load("/home/danai/Desktop/GaitTracking/p5/2.a/labels/2185.pt")

data_handler.print_data(img, label)
new_img, new_labels = data_handler.rotate(img, label, -20)
data_handler.print_data(new_img, new_labels)
new_img, new_labels = data_handler.mirror(*data_handler.rotate(img, label, 20))
data_handler.print_data(new_img, new_labels)
