import torch
from torch.utils.data import Dataset, DataLoader
import os
import random

random.seed(0)

class LegDataLoader():

    """expecting to find at data_paths a data and a labels folder"""
    def __init__(self, data_paths=["/home/athdom/GaitTracking/p1/2.a","/home/athdom/GaitTracking/p5/2.a", "/home/athdom/GaitTracking/p11/2.a", "/home/athdom/GaitTracking/p11/3.a", "/home/athdom/GaitTracking/p16/3.a", "/home/athdom/GaitTracking/p17/3.a", "/home/athdom/GaitTracking/p18/2.a", "/home/athdom/GaitTracking/p18/3.a"]):
        self.data_paths = data_paths

    def load(self, batch_size = 64):
        self.data = []
        for path in self.data_paths:
            self.data.append(sorted(os.listdir(path + "/data"), key = lambda a: int(a.split(".")[0])))
        self.batched_data = []
        self.batched_labels = []
        self.batch_size = batch_size
        i = batch_size
        batch_data = []
        batch_labels = []
        for vid_i, video in enumerate(self.data):
            prev_frame = None
            for frame in video:
                frame_i = int(frame.split(".")[0])
                if i == 0 or prev_frame and prev_frame + 1 != frame_i:
                    #print(i, prev_frame, frame_i)
                    self.batched_data.append(torch.stack(batch_data, dim = 0))
                    self.batched_labels.append(torch.stack(batch_labels, dim = 0))
                    batch_data = [torch.jit.load(self.data_paths[vid_i] + "/data/" + frame)]
                    batch_labels = [torch.jit.load(self.data_paths[vid_i] + "/labels/" + frame)]
                    i = batch_size - 1
                else:
                    batch_data.append(torch.jit.load(self.data_paths[vid_i] + "/data/" + frame))
                    batch_labels.append(torch.jit.load(self.data_paths[vid_i] + "/labels/" + frame))
                    i -= 1
                prev_frame = frame_i

        s = list(zip(self.batched_data, self.batched_labels))
        random.shuffle(s)
        self.batched_data, self.batched_labels = zip(*s)
        return self.batched_data[:int(len(self.batched_data) * 0.7)], self.batched_labels[:int(len(self.batched_labels) * 0.7)], self.batched_data[int(len(self.batched_data) * 0.7):int(len(self.batched_data) * 0.85)], self.batched_labels[int(len(self.batched_labels) * 0.7):int(len(self.batched_labels) * 0.85)], self.batched_data[int(len(self.batched_data) * 0.85):], self.batched_labels[int(len(self.batched_labels) * 0.85):]
