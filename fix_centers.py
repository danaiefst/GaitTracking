from matplotlib import pyplot as plt
import numpy as np
import torch
import os
import sys

person = sys.argv[1]
scenario = sys.argv[2]
path = sys.argv[3]

os.chdir("{}/{}/{}".format(path, person, scenario))
valid = open("valid.txt", "r")
centers = np.genfromtxt("valid_centers.csv", delimiter = ",")
max_height = 1.2
min_height = 0.1
max_width = 0.5
min_width = -0.5
img_side = 100
grid = 8
tags = []

for center in centers:
    y1 = (center[0] - min_width) / (max_width - min_width) * grid
    x1 = grid - (center[1] - min_height) / (max_height - min_height) * grid
    y2 = (center[2] - min_width) / (max_width - min_width) * grid
    x2 = grid - (center[3] - min_height) / (max_height - min_height) * grid
    tags.append([[int(x1), int(y1), x1 % 1, y1 % 1], [int(x2), int(y2), x2 % 1, y2 % 1]])

tags_i = 0
i = 0
for line in valid:
    start, end = line.strip().split(" ")
    i = int(start)
    end = int(end)
    while i <= end:
        torch.save(torch.tensor(tags[tags_i], dtype=torch.double), "labels/{}.pt".format(i))
        tags_i += 1
        i += 1
