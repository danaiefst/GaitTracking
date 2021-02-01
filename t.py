import numpy as np

laser = np.genfromtxt("/home/danai/Desktop/GaitTracking/data/cgdata/laserpoints.csv", delimiter = ",")

for i in laser:
    print(i)
