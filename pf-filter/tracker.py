import cv2
import numpy as np
import math
from filterpy import monte_carlo

class Tracker:
    def __init__(self, img, rect):
        self.sigx = 3
        self.sigy = 0.5
        self.n = 200
        self.nBins = 8
        self.lamb = 10
        self.patchWidth = rect[2]
        self.patchHeight = rect[3]
        self.xi = [rect[0] + rect[2] // 2, rect[1] + rect[3] // 2,]
        self.channels = 1
        if len(img.shape) == 3:
            self.channels = img.shape[2]
        self.desc = self.computeDescription(self.xi , img)

        #init particles
        self.xp = []
        for i in range(self.n):
            self.xp.append(self.xi)

        self.drawParticles(img, self.xi, self.xp)

        cv2.waitKey(0)

    def drawParticles(self, img, center, particles):
        print("center : ", center)
        imgDebug = cv2.circle(img, center, radius=0, color=(0, 0, 255), thickness=10)
        for part in particles:
            imgDebug = cv2.circle(imgDebug, (int(part[0]), int(part[1])), radius=0, color=(255, 0, 0), thickness=5)

        cv2.imshow("particles", imgDebug)
        print("particles")
        cv2.waitKey(10)

    def computeDescription(self, center, img):
        cd = np.zeros(self.nBins * self.channels)

        if center[0] < self.patchWidth // 2 or center[1] < self.patchHeight // 2 :
           return cd

        rect = [center[0] - self.patchWidth // 2, center[1] - self.patchHeight // 2, self.patchWidth, self.patchHeight]
        patch = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

        # Compute distance
        patchCenter = [patch.shape[0] // 2, patch.shape[1] // 2]
        dist = np.zeros(patch.shape[:2])
        for i in range(patch.shape[0]):
            for j in range(patch.shape[1]):
                dist[i, j] = math.sqrt((i - patchCenter[0])**2 + (j - patchCenter[1]) **2)
        dist /= np.amax(dist)


        # Get pixel value and add distance coeff in corresponding bin
        for i in range(patch.shape[0]):
            for j in range(patch.shape[1]):
                pixel = patch[i, j]
                for c in range(self.channels):
                    val = math.ceil(pixel[c] / 255 * (self.nBins - 1))
                    kE = 0
                    if dist[i, j]**2 < 1:
                        kE = 2 / math.pi * (1 - dist[i, j]**2)
                    cd[val + c*self.nBins] += kE

        return cd

    def updateParticles(self, xprev):
        return np.around(xprev + np.power([self.sigx, self.sigy], 2) * np.random.normal(size=len(xprev)))

    def bhattacharyya(self, newDesc):
        vec =  np.sum(np.sqrt(newDesc *self.desc))
        vec /= np.sum(self.desc) # not sure of this but more logical result
        return vec

    def track(self, img):
        xpPred = self.xp
        w = np.zeros(self.n)

        for i in range(self.n):
            coords = self.updateParticles(xpPred[i])
            desc = self.computeDescription([int(coords[0]), int(coords[1])], img)
            rho = self.bhattacharyya(desc)
            w[i] = math.exp(-self.lamb*(1 - rho)**2)
            xpPred[i] = coords

        if np.sum(w) > 0.01:
            w /= np.sum(w)

        outIdx = monte_carlo.residual_resample(w)
        self.xp = []
        for idx in outIdx:
            self.xp.append(xpPred[idx])

        xEstimate = np.around(np.matmul(w,self.xp)).astype('int32')
        xpPred = np.around(self.xp)

        self.drawParticles(img, xEstimate, xpPred)


        

