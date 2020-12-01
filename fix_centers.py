from matplotlib import pyplot as plt
import numpy as np
import torch
import os
import sys
from data_handler import transforml

person = sys.argv[1]
scenario = sys.argv[2]
source_path = sys.argv[3]
dest_path = sys.argv[4]

os.chdir("{}/{}/{}".format(source_path, person, scenario))
valid = open("valid.txt", "r")
centers = np.genfromtxt("centers.csv", delimiter = ",")
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

i = 0
last = len(centers)
for line in valid:
    start, end = line.strip().split(" ")
    i = int(start)
    end = int(end)
    while i <= end:
        print(i)
        if(tags[i][0][0] >= 8 or tags[i][0][1] >= 8 or tags[i][1][0] >= 8 or tags[i][1][1] >= 8):
            print("Error", i)
        torch.save(torch.tensor(tags[i], dtype=torch.double), "{}/{}/{}/labels/{}.pt".format(dest_path, person, scenario, i))
        transformed = transforml(tags[i])
        for j in range(len(transformed)):
            if(transformed[j][0][0] >= 8 or transformed[j][0][1] >= 8 or transformed[j][1][0] >= 8 or transformed[j][1][1] >= 8):
                print("Error transformed", i, j)
            torch.save(transformed[j], "{}/{}/{}/labels/{}.pt".format(dest_path, person, scenario, j * (last + 1) + i))
        i += 1
