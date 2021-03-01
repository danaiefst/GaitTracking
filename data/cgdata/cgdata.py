from matplotlib import pyplot as plt
import numpy as np
from time import sleep
import torch
import os

np.random.seed(0)

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
        self.thetax += (self.dt + (np.random.random() - 0.5) * 0.5 * self.dt) * accel
        self.thetay += self.dt * accel
        self.x = self.offsetx + self.ampx * np.sin(self.thetax / self.delayx)
        self.y = self.offsety + self.ampy * np.sin(self.thetay / self.delayy)


class walk:

    def __init__(self, rleg=None, lleg=None, direction=0, centeroffsetx=None, centeroffsety=None, maxinterdist=0.3, mininterdist=0.1, laser_range=np.pi, resolution=512, maxbema=0.4, minbema=0.1, dt=1/40, occlusions_only=False, show_trails=False):
        self.show_trails = show_trails
        self.occlusions_only = occlusions_only
        if not occlusions_only:
            self.direction = direction
        else:
            theta_a = 28 * np.pi / 180
            theta_b = 38 * np.pi / 180
            if np.random.random() < 0.5:
                self.direction = np.random.random() * (theta_b - theta_a) + theta_a
            else:
                self.direction = np.random.random() * (theta_a - theta_b) - theta_a
        self.maxinterdist = maxinterdist
        self.mininterdist = mininterdist
        if centeroffsetx is None:
            self.centeroffsetx = (MINX + MAXX) / 2 + (np.random.random() - 0.5) * (MAXX - MINX) / 8
        else:
            self.centeroffsetx = centeroffsetx
        if centeroffsety is None:
            self.centeroffsety = np.random.random() * (MAXY - 2 * minbema - MINY) / 2 + (MAXY - MINY) / 2
        else:
            self.centeroffsety = centeroffsety
        if rleg is None:
            rradius = np.random.random() * 0.02 + 0.08
            lradius = rradius + (np.random.random() - 0.5) * 0.02
            self.interdist = np.random.random() * (maxinterdist - mininterdist) + mininterdist
            self.centeroffsety += (np.random.random() - 0.5) * 0.2
            self.centeroffsetx += (np.random.random() - 0.5) * 0.01
            bema = np.random.random() * (min(maxbema, max(MAXY - self.centeroffsety, self.centeroffsety - MINY)) - minbema) + minbema
            bemax = (np.random.random() - 0.5) * bema / 5
            self.rleg = leg(rradius, self.centeroffsetx - self.interdist / 2, self.centeroffsety, bemax, bema/2, 0, 0, np.random.random() * 2 + 1, 1, dt)
            self.lleg = leg(lradius, self.centeroffsetx + self.interdist / 2, self.centeroffsety, bemax, bema/2, 0, np.pi, np.random.random() + 1, 1, dt)
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
        self.have_trails = [False, False, False, False]
        self.trail_size = np.random.random() * 0.15 + 0.15

    def move(self):
        for i in range(4):
            if self.have_trails[i]:
                if np.random.random() < 0.1:
                    self.have_trails[i] = False
            else:
                if np.random.random() < 0.05:
                    self.have_trails[i] = True
        if not self.occlusions_only:
            self.statetimer += 1
            if self.stateend == self.statetimer:
                if np.random.random() < 0.8:
                    self.state = 1
                else:
                    self.state = 0
                self.statetimer = 0
                self.stateend = np.random.randint(50, 200)
                self.angle = 0
                self.startangle = self.direction
                if self.state != 0:
                    xM, xm = self.lleg.ampx + self.lleg.offsetx + self.lleg.radius, -self.rleg.ampx + self.rleg.offsetx - self.rleg.radius
                    yMl, yml = self.lleg.ampy + self.lleg.offsety + self.lleg.radius, -self.lleg.ampy + self.lleg.offsety - self.lleg.radius
                    yMr, ymr = self.rleg.ampy + self.rleg.offsety + self.rleg.radius, -self.rleg.ampy + self.rleg.offsety - self.rleg.radius
                    #legacy rotation
                    #maxangle = min(np.pi / 2 + np.arctan(np.sqrt(xm ** 2 + yMr ** 2 - MINX ** 2) / MINX), np.pi / 2 + np.arctan(MINY / -np.sqrt(xM ** 2 + ymr ** 2 - MINY ** 2)))
                    #minangle = max(np.arctan(np.sqrt(xM ** 2 + yMl ** 2 - MAXX ** 2) / MAXX) - np.pi / 2, np.arctan(MINY / np.sqrt(xM ** 2 + yml ** 2 - MINY ** 2)) - np.pi / 2)
                    maxangle = 35 * np.pi / 180 
                    minangle = -35 * np.pi / 180 
                    self.angle = np.random.random() * (maxangle - minangle) + minangle
            self.direction += (self.angle - self.startangle) / self.stateend
        self.rleg.move()
        self.lleg.move()

    def _rotate(self, x, y, theta):
        x0, y0 = self.centeroffsetx, self.centeroffsety
        return ((x - x0) * np.cos(theta) - (y - y0) * np.sin(theta) + x0, (y - y0) * np.cos(theta) + (x - x0) * np.sin(theta) + y0)
    
    def _coords(self):
        """Returns the coords of the two legs as tuples. Return value of the form ((rlegx, rlegy), (llegx, llegy))"""
        return (self._rotate(self.rleg.x, self.rleg.y, self.direction), self._rotate(self.lleg.x, self.lleg.y, self.direction))

    def laser_points(self, noise=0.003, vision_ratio=0.8, vision_ratio2=0.83):
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
                elif (abs(x1 - a[i] * y1) / np.sqrt(a[i] ** 2 + 1)) / self.rleg.radius < vision_ratio2:
                    if self.show_trails:
                        tempyp = (-B1[i] - np.sqrt(D1[i])) / 2 / A1[i]
                        which_trail = int(np.arctan(a[i]) < np.arctan(x1 / y1))
                        if self.have_trails[which_trail]:
                            phi2 = np.arctan((x1 - a[i] * tempyp) / (y1 - tempyp))
                            phi1 = np.arctan(a[i])
                            phi = abs(phi2 - phi1)
                            if phi > np.pi / 2:
                                phi = np.pi - phi
                            bounceoffset = max(min(np.tan(phi) / 8 + (np.random.random() - 0.5), self.trail_size), 0)
                            yp = min(bounceoffset + tempyp, yp)
            if D2[i] >= 0:
                if (abs(x2 - a[i] * y2) / np.sqrt(a[i] ** 2 + 1)) / self.lleg.radius < vision_ratio:
                    yp = min(yp, (-B2[i] - np.sqrt(D2[i])) / 2 / A2[i]) + np.random.normal(scale=noise) * np.sin(theta[i])
                elif (abs(x2 - a[i] * y2) / np.sqrt(a[i] ** 2 + 1)) / self.lleg.radius < vision_ratio2:
                    if self.show_trails:
                        tempyp = min(yp, (-B2[i] - np.sqrt(D2[i])) / 2 / A2[i]) + np.random.normal(scale=noise) * np.sin(theta[i])
                        which_trail = int(np.arctan(a[i]) < np.arctan(x2 / y2))
                        if self.have_trails[which_trail + 2]:
                            phi2 = np.arctan((x2 - a[i] * tempyp) / (y2 - tempyp))
                            phi1 = np.arctan(a[i])
                            phi = abs(phi2 - phi1)
                            if phi > np.pi / 2:
                                phi = np.pi - phi
                            bounceoffset = max(min(np.tan(phi) / 8 + (np.random.random() - 0.5), self.trail_size), 0)
                            yp = min(bounceoffset + tempyp, yp)
            if yp == np.inf or yp == -np.inf:
                yp = 0
            y.append(yp)
            x.append(a[i] * yp)
        return np.array(x), np.array(y)

    def coords(self, offset=0.06):
        (x1, y1), (x2, y2) = self._coords()
        u = np.arctan2(y1, x1)
        x1 -= offset * np.cos(u)
        y1 -= offset * np.sin(u)
        u = np.arctan2(y2, x2)
        x2 -= offset * np.cos(u)
        y2 -= offset * np.sin(u)
        return (x1, y1), (x2, y2)

    def gait_state(self):
        """
        Returns the state of the gait as an index:
        1: RDS (left double support (left leg in front of right leg))
        2: RS
        3: LDS
        4: LS
        """
        theta = ((self.rleg.thetay + np.pi / 2) % (2 * np.pi)) / (2 * np.pi)
        for i, e in enumerate([0.15, 0.45, 0.6, 1]):
            if theta < e:
                return (i + 2) % 4 + 1
        

#plt.ion()
#fig, ax = plt.subplots()

#plt.xlim(MINX, MAXX)
#plt.ylim(MINY, MAXY)
#r, = ax.plot([0], [0], 'o')
#l, = ax.plot([0], [0], 'o')
#ls, = ax.plot([0], [0], 'o', markersize=1)
N = 1000
vision = 0.6
c = 0
valid = open("valid.txt", "w")
laser = np.zeros((13 * 4 * N, 1024))
centers = np.zeros((13 * 4 * N, 4))
#gait_states = np.zeros((13 * 4 * N))
for accel in [2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5]:
    for j in range(3):
        print("{} {}".format(c, c + N - 1), file=valid)
        W = walk(show_trails=True)
        for i in range(N):
            rd, ld = W.coords()
            #r.set_data([rd[0]], [rd[1]])
            #l.set_data([ld[0]], [ld[1]])
            lx, ly = W.laser_points(vision_ratio=vision, noise=0.002)
            #ls.set_data(lx, ly)
            #fig.canvas.draw()
            #fig.canvas.flush_events()
            #sleep(1/40)
            W.move()
            laser[c] = np.stack([lx, ly]).T.reshape(-1)
            centers[c, 0], centers[c, 1], centers[c, 2], centers[c, 3] = rd[0], rd[1], ld[0], ld[1]
            #gait_states[c] = W.gait_state()
            #print(gait_states[c])
            c += 1
    print("{} {}".format(c, c + N - 1), file=valid)
    W = walk()
    for i in range(N):
        rd, ld = W.coords()
        #r.set_data([rd[0]], [rd[1]])
        #l.set_data([ld[0]], [ld[1]])
        lx, ly = W.laser_points(vision_ratio=vision, noise=0.002)
        #ls.set_data(lx, ly)
        #fig.canvas.draw()
        #fig.canvas.flush_events()
        #sleep(1/40)
        W.move()
        laser[c] = np.stack([lx, ly]).T.reshape(-1)
        centers[c, 0], centers[c, 1], centers[c, 2], centers[c, 3] = rd[0], rd[1], ld[0], ld[1]
        #gait_states[c] = W.gait_state()
        #print(gait_states[c])
        c += 1
        
 
np.savetxt("laserpoints.csv", laser, delimiter=",")
np.savetxt("centers.csv", centers, delimiter=",")
#np.savetxt("gait_states.csv", gait_states, delimiter=",")
