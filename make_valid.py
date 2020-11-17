import scipy.io
from matplotlib import pyplot as plt
from time import sleep
import numpy as np
import os

os.chdir("/home/danai/Desktop/mobot/p18/3.a")
centers = open("centers.csv", "r")
laser = open("laserpoints.csv", "r")
new_centers = open("valid_centers.csv", "w")
new_laser = open("valid_laserpoints.csv", "w")
valid = open("valid.txt", "r")

line_i = 0
for line in valid:
    start, end = line.strip().split(",")
    start = int(start)
    end = int(end)
    while line_i < start:
        centers.readline()
        laser.readline()
        line_i += 1
    while line_i <= end:
        center = centers.readline()
        l = laser.readline()
        new_centers.write(center)
        new_laser.write(l)
        line_i += 1

new_centers.close()
new_laser.close()
