import torch
import os
import numpy as np

path = "data/p17/3.a"

os.chdir(path)
states = np.genfromtxt("gait_states.csv", delimiter=",")

for i in range(len(states)):
    states[i] = (states[i] - 3) % 4 + 1

np.savetxt("gait_states.csv", states, delimiter=",")
