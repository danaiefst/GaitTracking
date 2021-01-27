from matplotlib import pyplot as plt
import numpy as np
from time import sleep
import torch
import os

MAXX = 0.5
MINX = -0.5
MAXY = 1.2
MINY = 0.2
img_side = 112

accel = 8

class leg:

    def __init__(self, radius, offsetx, offsety, ampx, ampy, thetax, thetay, delayx, delayy, dt, accelx=1, accely=1):
        self.offsetx = offsetx
        self.offsety = offsety
        self.ampx = ampx
        self.ampy = ampy
        self.delayx = delayx
        self.delayy = delayy
        self.thetax = thetax
        self.thetay = thetay
        self.accelx = accelx
        self.accely = accely
        self.dt = dt
        self.radius = radius
        self.x = self.offsetx + self.ampx * np.sin(self.thetax / delayx)
        self.y = self.offsety + self.ampy * np.sin(self.thetay / delayy)

    def move(self):
        self.thetax += self.dt * accel
        self.thetay += self.dt * accel
        self.x = self.offsetx + self.ampx * np.sin(self.thetax / self.delayx)
        self.y = self.offsety + self.ampy * np.sin(self.thetay / self.delayy)


class walk:

    def __init__(self, rleg=None, lleg=None, direction=0, centeroffsetx=None, centeroffsety=None, maxinterdist=0.3, mininterdist=0.1, laser_range=np.pi, resolution=700, maxbema=1, minbema=0.2, dt=1/40):
        self.direction = direction
        self.maxinterdist = maxinterdist
        self.mininterdist = mininterdist
        if centeroffsetx is None:
            self.centeroffsetx = (MINX + MAXX) / 2
        else:
            self.centeroffsetx = centeroffsetx
        if centeroffsety is None:
            self.centeroffsety = (MINY + MAXY) / 2
        else:
            self.centeroffsety = centeroffsety
        if rleg is None:
            rradius = np.random.random() * 0.02 + 0.08
            lradius = rradius + (np.random.random() - 0.5) * 0.02
            self.interdist = np.random.random() * (maxinterdist - mininterdist) + mininterdist
            self.centeroffsety += (np.random.random() - 0.5) * 0.2
            self.centeroffsetx += (np.random.random() - 0.5) * 0.01
            bema = np.random.random() * (min(maxbema, max(MAXY - self.centeroffsety, self.centeroffsety - MINY)) - minbema) + minbema
            self.rleg = leg(rradius, self.centeroffsetx - self.interdist / 2, self.centeroffsety, bema/40, bema/2, 0, 0, 2, 1, dt)
            self.lleg = leg(lradius, self.centeroffsetx + self.interdist / 2, self.centeroffsety, bema/40, bema/2, 0, np.pi, 2, 1, dt)
        else:
            self.rleg = rleg
            self.lleg = lleg
            self.interdist = lleg.x - rleg.x
        self.laser_range = laser_range
        self.resolution = resolution
        self.state = 0
        self.statetimer = 0
        self.startangle = 0
        self.angle = 0
        self.stateend = np.random.randint(50, 200)

    def move(self):
        self.statetimer += 1
        if self.stateend == self.statetimer:
            self.state = np.random.randint(0, 2)
            self.statetimer = 0
            self.stateend = np.random.randint(50, 200)
            self.angle = 0
            self.startangle = self.direction
            if self.state != 0:
                xM, xm = self.lleg.ampx + self.lleg.offsetx + self.lleg.radius, -self.rleg.ampx + self.rleg.offsetx - self.rleg.radius
                yMl, yml = self.lleg.ampy + self.lleg.offsety + self.lleg.radius, -self.lleg.ampy + self.lleg.offsety - self.lleg.radius
                yMr, ymr = self.rleg.ampy + self.rleg.offsety + self.rleg.radius, -self.rleg.ampy + self.rleg.offsety - self.rleg.radius
                maxangle = min(np.pi / 2 + np.arctan(np.sqrt(xm ** 2 + yMr ** 2 - MINX ** 2) / MINX), np.pi / 2 + np.arctan(MINY / -np.sqrt(xM ** 2 + ymr ** 2 - MINY ** 2)))
                minangle = max(np.arctan(np.sqrt(xM ** 2 + yMl ** 2 - MAXX ** 2) / MAXX) - np.pi / 2, np.arctan(MINY / np.sqrt(xM ** 2 + yml ** 2 - MINY ** 2)) - np.pi / 2)
                self.angle = np.random.random() * (maxangle - minangle) + minangle
        self.direction += (self.angle - self.startangle) / self.stateend
        self.rleg.move()
        self.lleg.move()

    @staticmethod
    def _rotate(x, y, theta):
        return (x * np.cos(theta) - y * np.sin(theta), y * np.cos(theta) + x * np.sin(theta))
    
    def _coords(self):
        """Returns the coords of the two legs as tuples. Return value of the form ((rlegx, rlegy), (llegx, llegy))"""
        return (self._rotate(self.rleg.x, self.rleg.y, self.direction), self._rotate(self.lleg.x, self.lleg.y, self.direction))

    def laser_points(self, noise=0.002, vision_ratio=0.8):
        (x1, y1), (x2, y2) = self._coords()
        theta = np.linspace(np.pi/2 - self.laser_range / 2, np.pi/2 + self.laser_range / 2, self.resolution)
        a = np.cos(theta) / np.sin(theta)
        A1 = a ** 2 + 1
        B1 = -2 * y1 - 2 * x1 * a
        C1 = y1 ** 2 + x1 ** 2 - self.rleg.radius ** 2
        D1 = B1 ** 2 - 4 * A1 * C1
        A2 = a ** 2 + 1
        B2 = -2 * y2 - 2 * x2 * a
        C2 = y2 ** 2 + x2 ** 2 - self.rleg.radius ** 2
        D2 = B2 ** 2 - 4 * A2 * C2
        y = []
        x = []
        for i in range(len(D1)):
            yp = np.inf
            if D1[i] >= 0:
                if (abs(x1 - a[i] * y1) / np.sqrt(a[i] ** 2 + 1)) / self.rleg.radius < vision_ratio:
                    yp = (-B1[i] - np.sqrt(D1[i])) / 2 / A1[i] + np.random.normal(scale=noise) * np.sin(theta[i])
            if D2[i] >= 0:
                if (abs(x2 - a[i] * y2) / np.sqrt(a[i] ** 2 + 1)) / self.lleg.radius < vision_ratio:
                    yp = min(yp, (-B2[i] - np.sqrt(D2[i])) / 2 / A2[i]) + np.random.normal(scale=noise) * np.sin(theta[i])
            y.append(yp)
            x.append(a[i] * yp)
        return np.array(x), np.array(y)

    def coords(self, offset=0.04):
        (x1, y1), (x2, y2) = self._coords()
        u = np.arctan2(y1, x1)
        x1 -= offset * np.cos(u)
        y1 -= offset * np.sin(u)
        u = np.arctan2(y2, x2)
        x2 -= offset * np.cos(u)
        y2 -= offset * np.sin(u)
        return (x1, y1), (x2, y2)

data_path = "/home/danai/Desktop/GaitTracking/cgdata"
N = 1000
vision = 0.7
W = walk()
#plt.ion()
#fig, ax = plt.subplots()

#plt.xlim(MINX, MAXX)
#plt.ylim(MINY, MAXY)
#r, = ax.plot([0], [0], 'o')
#l, = ax.plot([0], [0], 'o')
#laser, = ax.plot([0], [0], 'o', markersize=1)
c = -1
for accel in [2, 3, 4, 5]:
    c += 1
    for i in range(1000):
        rd, ld = W.coords()
        #r.set_data([rd[0]], [rd[1]])
        #l.set_data([ld[0]], [ld[1]])
        lx, ly = W.laser_points(vision_ratio = vision, noise = 0.002)
        #laser.set_data(lx, ly)
        #fig.canvas.draw()
        #fig.canvas.flush_events()
        #sleep(1/40)
        W.move()

        img = torch.zeros((img_side, img_side), dtype=torch.double)
        valid = np.where(ly != np.inf)
        y = (lx[valid] - MINX) / (MAXX - MINX) * img_side
        x = img_side - (ly[valid] - MINY) / (MAXY - MINY) * img_side
        img[x.astype(int), y.astype(int)] = 1
        torch.save(img, "{}/data/{}.pt".format(data_path, c))
        y1 = (rd[0] - MINX) / (MAXX - MINX)
        x1 = 1 - (rd[1] - MINY) / (MAXY - MINY)
        y2 = (ld[0] - MINX) / (MAXX - MINX)
        x2 = 1 - (ld[1] - MINY) / (MAXY - MINY)
        torch.save(torch.tensor([[x1, y1], [x2, y2]], dtype=torch.double), "{}/labels/{}.pt".format(data_path, c))
        c += 1

