#Copyright 2018 Dennis Schunder
#
#Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
#
#http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT #WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy as np
import h5py
from math import *
#import utils_find_1st as utf1st
import random
import pyqtgraph as pg
import pyqtgraph.opengl as gl


class OctTree(object):
    parent = None
    groupPoints = None  # array of (t, points), with t being the group number
    corners = None
    groups = None
    eps = None
    avg = None
    depth = None
    max_distance = None
    children = None
    depthBounds = None
    startStep = -1
    lastStep = -1
    size = -1

    def __getstate__(self):
        save = {}

        for x in self.__dict__:
            if x == 'similarity':
                if not self.similarity is None:
                    save[x] = self.similarity.tolist()
                continue
            save[x] = self.__dict__[x]
        return save

    def __setstate__(self, state):
        for x in state:
            if x == 'similarity':
                self.similarity = np.array(state[x])
                continue
            self.__dict__[x] = state[x]
        self.similarity = None

    def __init__(self, groupPoints, depthBounds, eps, parent=None, corners=(), depth=0):
        #self.parent = parent
        self.groupPoints = groupPoints
        self.depth = depth
        self.depthBounds = depthBounds
        self.eps = eps
        self.similarity = None
        self.simRange = None

        # No corners given, calculate them across groups
        if len(corners) == 0:
            self.calcCorners()
        else:
            self.corners = corners
            #print(" -> Set corners as", corners)

        # Determine current size
        self.size = self.getSize()

        print(" " * depth, "%d: [%d]" % (depth, self.size))

        # Ignore empty trees
        if self.size == 0:
            return

        if self.depth < self.depthBounds[0]:
            self.divide()
            return

        # Calculate average vector
        self.calcAverage()

        # Check if maximum depthBounds reached
        if self.depth >= depthBounds[1]:
            return

        #self.max_distance = sqrt((pow(self.corners[0] - self.corners[1], 2)) + (pow(self.corners[2] - self.corners[3], 2)))

        if not self.isSimilar():
            self.divide()
            return

    def calcCorners(self):
        xMin = 0
        xMax = 0
        yMin = 0
        yMax = 0
        zMin = self.groupPoints[0][0]
        zMax = self.groupPoints[-1][0]

        # Find boundaries across all groups
        for group in self.groupPoints:
            (X, Y) = group[1][1]

            gxmin = np.amin(X)
            gxmax = np.amax(X)
            gymin = np.amin(Y)
            gymax = np.amax(Y)

            if gxmin < xMin:
                xMin = gxmin
            if gxmax > xMax:
                xMax = gxmax
            if gymin < yMin:
                yMin = gymin
            if gymax > yMax:
                yMax = gymax

        self.corners = (xMin, xMax, yMin, yMax, zMin, zMax)
        #print(" -> Determining corners...", self.corners)

    def calcAverage(self):
        xList = []
        yList = []
        uList = []
        vList = []

        # Get list of averages per group
        for group in self.groupPoints:
            (X, Y) = group[1][1]
            (u, v) = group[1][2]
            xList.append(np.average(X))
            yList.append(np.average(Y))
            uList.append(np.average(u))
            vList.append(np.average(v))

        # Calculate actual averages
        xAvg = np.average(xList)
        yAvg = np.average(yList)
        uAvg = np.average(uList)
        vAvg = np.average(vList)
        groupAvg = (self.corners[5] + self.corners[4]) / 2

        self.avg = ((xAvg, yAvg, groupAvg), (uAvg, vAvg))

    def divide(self):
        (xMid, yMid, zMid) = (
            np.average(self.corners[0:2]),
            np.average(self.corners[2:4]),
            np.average(self.corners[4:6]))

        ul = (self.corners[0],            xMid,
              self.corners[2], yMid,
              zMid, self.corners[5])
        ur = (xMid, self.corners[1], self.corners[2], yMid, zMid, self.corners[5])
        bl = (self.corners[0],            xMid,
              yMid, self.corners[3],
              self.corners[4], zMid)
        br = (xMid, self.corners[1], yMid, self.corners[3], self.corners[4], zMid)

        midIndex = ceil(zMid) + 1

        frontGroups = self.groupPoints[:midIndex]
        backGroups = self.groupPoints[midIndex:]

        frontVoxels = self.getVoxels(frontGroups, (xMid, yMid))
        backVoxels = self.getVoxels(backGroups, (xMid, yMid))

        # Erase points so only children hold data
        self.groupPoints = None

        Q1 = OctTree(frontVoxels[0], self.depthBounds,
                     self.eps, self, ul, self.depth + 1)
        Q2 = OctTree(frontVoxels[1], self.depthBounds,
                     self.eps, self, ur, self.depth + 1)
        Q3 = OctTree(frontVoxels[2], self.depthBounds,
                     self.eps, self, bl, self.depth + 1)
        Q4 = OctTree(frontVoxels[3], self.depthBounds,
                     self.eps, self, br, self.depth + 1)
        Q5 = OctTree(backVoxels[0], self.depthBounds,
                     self.eps, self, ul, self.depth + 1)
        Q6 = OctTree(backVoxels[1], self.depthBounds,
                     self.eps, self, ur, self.depth + 1)
        Q7 = OctTree(backVoxels[2], self.depthBounds,
                     self.eps, self, bl, self.depth + 1)
        Q8 = OctTree(backVoxels[3], self.depthBounds,
                     self.eps, self, br, self.depth + 1)

        self.children = [Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8]

    def isSimilar(self, fullCheck=False):
        avgVel = self.avg[1]
        avgNorm = np.linalg.norm(avgVel)

        similarities = np.empty(self.getSize(), dtype='uint32, f8')

        i = 0

        for group in self.groupPoints:
            ids = group[1][0]
            pos = group[1][1]
            vel = group[1][2]

            velNorm = np.linalg.norm(vel, axis=0)

            normProd = np.multiply(avgNorm, velNorm)

            dotProd = np.dot(avgVel, vel)

            #Prevent division by 0
            invInds = np.where(normProd == 0)[0]
            replaceInds = invInds + i

            invLength = len(invInds)

            if invLength != 0:
                replace = np.zeros(invLength, dtype='uint32, f8')
                replace['f0'] = ids[invInds]

                similarities[replaceInds] = replace
                
                i = i + invLength
                normProd = np.delete(normProd, invInds)
                dotProd = np.delete(dotProd, invInds)

                if len(normProd) == 0:
                    continue

            cosSim = np.divide(dotProd, normProd)

            # Fix invalid values
            cosSim[cosSim < -1] = -1
            cosSim[cosSim > 1] = 1

            similarity = np.divide(cosSim+1, 2)

            #dist = np.divide(np.arccos(cosSim), np.pi) # Angular distance
            #similarity = np.subtract(1, dist)

            #Attach IDs to the similarity
            sortInds = np.argsort(similarity)
            elementCount = len(sortInds)

            if elementCount == 0:
                continue

            endInd = elementCount + i

            groupSimilarity = np.empty(elementCount, dtype='uint32, f8')
            groupSimilarity['f0'] = ids[sortInds]
            groupSimilarity['f1'] = similarity[sortInds]

            similarities[i:endInd] = groupSimilarity
            i = endInd

            # Check if element of lowest similarity is still similar enough
            if not fullCheck and groupSimilarity['f1'][0] < self.eps:
                similarities = None
                return False

        self.similarity = similarities[np.argsort(similarities['f1'])]

        self.simRange = (self.similarity['f1'][0], self.similarity['f1'][-1])

        # All points were similar enough
        return True

    def getVoxels(self, groups, midPoints):

        upLeftVoxel = []
        upRightVoxel = []
        botLeftVoxel = []
        botRightVoxel = []

        for (t, data) in groups:

            ids = data[0]
            (X, Y) = data[1]
            (u, v) = data[2]

            upInd = Y < midPoints[1]
            botInd = np.logical_not(upInd)
            leftInd = X < midPoints[0]
            rightInd = np.logical_not(leftInd)

            upLeftInd = np.logical_and(upInd, leftInd)
            upRightInd = np.logical_and(upInd, rightInd)
            botLeftInd = np.logical_and(botInd, leftInd)
            botRightInd = np.logical_and(botInd, rightInd)

            upLeftVoxel.append((t, (ids[upLeftInd], (X[upLeftInd], Y[upLeftInd]), (u[upLeftInd], v[upLeftInd]))))
            upRightVoxel.append((t, (ids[upRightInd], (X[upRightInd], Y[upRightInd]), (u[upRightInd], v[upRightInd]))))
            botLeftVoxel.append((t, (ids[botLeftInd], (X[botLeftInd], Y[botLeftInd]), (u[botLeftInd], v[botLeftInd]))))
            botRightVoxel.append((t, (ids[botRightInd], (X[botRightInd], Y[botRightInd]), (u[botRightInd], v[botRightInd]))))

        
        return (upLeftVoxel, upRightVoxel, botLeftVoxel, botRightVoxel)

    def getStartStep(self):
        return self.corners[4]

    def getLastStep(self):
        return self.corners[5]

    def getSize(self):
        # Don't recalculate size
        if self.size != -1:
            return self.size

        size = 0

        for group in self.groupPoints:
            size = size + len(group[1][0])

        return size

    def getVector(self):
        if self.size == 0:
            return []

        if self.children != None:
            reps = []

            # Recursively try to find the children
            for child in self.children:
                reps = reps + child.getVector()
        else:
            # Find the representative vector based on closest similarity

            maxSim = -1

            if self.similarity is None or self.similarity['f1'] is None:
                self.isSimilar(fullCheck=True)

            maxSim = self.similarity['f1'][-1]

            #Highest similarity has been found, get all elements with this similarity

            candidates = self.similarity[self.similarity['f1'] == maxSim]['f0'].tolist()

            if len(candidates) > 1:
                reps = [candidates[0]]
            elif len(candidates) == 1:
                reps = candidates
            else:
                print("Error ocurred at finding representative")
                return candidates

        return reps

    def findSimilarityRange(self):
        if self.children != None:

            minVal = -1
            maxVal = -1

            for child in self.children:
                if child.getSize() == 0:
                    continue

                (l, h) = child.findSimilarityRange()

                if minVal == -1 or l < minVal:
                    minVal = l
                if h > maxVal:
                    maxVal = h

            return (minVal, maxVal)
        else:
            if self.similarity is None or self.similarity['f1'] is None:
                self.isSimilar(fullCheck=True)
            if self.simRange is None:
                return (0, 0)
            else:
                return self.simRange

    def findSizeRange(self):
        if self.children != None:

            minVal = -1
            maxVal = -1

            for child in self.children:
                (l, h) = child.findSizeRange()

                if minVal == -1 or l < minVal:
                    minVal = l
                if h > maxVal:
                    maxVal = h

            return (minVal, maxVal)
        else:
            return (self.size, self.size)

    def countLeaves(self):
        if self.children != None:
            s = 0
            for child in self.children:
                s = s + child.countLeaves()
            return s
        else:
            return 1

    def getPath(self, chosen, path=None, offset=0):
        if self.depth == 0:
            size = self.corners[5] - self.corners[4] + 1
            path = np.zeros((4, size))
            path[2] = np.full(size, -1)
            offset = self.corners[4]

        if self.children != None:
            for child in self.children:
                child.getPath(chosen, path, offset)
        else:
            self.find(chosen, path, offset)

        if self.depth == 0:
            return path

    def find(self, pid, path, offset):
        if self.size == 0:
            return

        for group in self.groupPoints:
            step = group[0]

            (ids, pos, vel) = group[1]

            if len(ids) == 0:
                continue

            index = np.argmax(ids == pid)

            # np.argmax returns 0 if element was not found, check if 0 was the correct index
            if index == 0 and ids[0] != pid:
                continue

            pathIndex = step - offset

            path[:, pathIndex] = (pos[0][index], pos[1][index], pathIndex, self.size)

    def draw(self, pw, stepOffset, drawBounds=True, drawSlices=True, drawInner=False, drawZ=False):
        if self.size == 0:
            return

        zStart = self.corners[4] - stepOffset
        zEnd = self.corners[5] - stepOffset

        if self.depth == 0:
            xSize = self.corners[1] - self.corners[0]
            ySize = self.corners[3] - self.corners[2]
            zSize = self.corners[5] - self.corners[4]

            # Draw bounding box
            if drawBounds:
                b = gl.GLBoxItem(color=(255,255,255,255))

                b.setSize(xSize, ySize, zSize)
                pw.addItem(b)

                b.translate(self.corners[0], self.corners[2], zStart)

            if drawSlices:
                for i in range(zSize):
                    box = gl.GLBoxItem(color=(200,200,200,255))
                    box.setSize(xSize, ySize, 0)
                    box.translate(self.corners[0], self.corners[2], i)
                    pw.addItem(box)

        if self.children != None:

            if drawInner:
                if drawZ:
                    self.drawInner(pw, stepOffset)
                else:
                    self.drawInnerMarks(pw, stepOffset)

            for child in self.children:
                child.draw(pw, stepOffset, drawBounds, drawSlices, drawInner, drawZ)

    def drawInnerFrame(self, view, stepOffset):
        zStart = self.corners[4] - stepOffset
        zEnd = self.corners[5] - stepOffset

        xSize = self.corners[1] - self.corners[0]
        ySize = self.corners[3] - self.corners[2]
        zVals = (zStart, zEnd)

        for z in zVals:
            box = gl.GLBoxItem()
            box.setSize(xSize, ySize, 0)
            box.translate(0, 0, z)
            view.addItem(box)

    def drawFaces(self, view, stepOffset):
        zStart = self.corners[4] - stepOffset
        zEnd = self.corners[5] - stepOffset

        (xmin, xmax) = self.corners[0:2]
        (ymin, ymax) = self.corners[2:4]
        (zmin, zmax) = (zStart, zEnd)

        verts = np.array([
            [xmin, ymin, zmin],

            [xmax, ymin, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],

            [xmin, ymax, zmax],
            [xmax, ymax, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmin],
        ])
        faces = np.array([
            [0, 1, 3],
            [1, 3, 6],

            [1, 7, 5],
            [6, 7, 5],

            [4, 5, 6],
            [3, 4, 6],

            [2, 4, 5],
            [2, 5, 7],

            [0, 2, 4],
            [0, 3, 4],

            [0, 2, 1],
            [7, 2, 1]
        ])

        color = np.append(np.random.random(3), 0.5)

        mesh = gl.GLMeshItem(vertexes=verts, color=color, faces=faces, smooth=False,
                             drawFaces=True, drawEdges=False)
        view.addItem(mesh)
    

    def drawInner(self, view, stepOffset):
        zStart = self.corners[4] - stepOffset
        zEnd = self.corners[5] - stepOffset

        xmid = np.average(self.corners[0:2])
        ymid = np.average(self.corners[2:4])
        zmid = (zStart + zEnd) / 2

        (xmin, xmax) = self.corners[0:2]
        (ymin, ymax) = self.corners[2:4]
        (zmin, zmax) = (zStart, zEnd)

        verts = np.array([
            [xmin, ymid, zmid],
            [xmax, ymid, zmid],
            [xmid, ymin, zmid],
            [xmid, ymax, zmid],
            [xmid, ymid, zmin],
            [xmid, ymid, zmax],
            [xmin, ymid, zmin],
            [xmax, ymid, zmin],
            [xmin, ymin, zmid],
            [xmax, ymin, zmid],
            [xmid, ymin, zmin],
            [xmid, ymax, zmin],
            [xmin, ymin, zmid],
            [xmin, ymax, zmid],
            [xmid, ymin, zmin],
            [xmid, ymin, zmax],
            [xmin, ymid, zmin],
            [xmin, ymid, zmax],
            [xmin, ymid, zmax],
            [xmax, ymid, zmax],
            [xmin, ymax, zmid],
            [xmax, ymax, zmid],
            [xmid, ymin, zmax],
            [xmid, ymax, zmax],
            [xmax, ymin, zmid],
            [xmax, ymax, zmid],
            [xmid, ymax, zmin],
            [xmid, ymax, zmax],
            [xmax, ymid, zmin],
            [xmax, ymid, zmax],
            [xmax, ymin, zmid],
            [xmin, ymax, zmid],
            [xmid, ymin, zmax],
            [xmid, ymax, zmin],
        ])
        faces = np.array([
            [0, 1, 1],
            [2, 3, 3],
            [4, 5, 5],
            [6, 7, 7],
            [8, 9, 9],
            [10, 11, 11],
            [12, 13, 13],
            [14, 15, 15],
            [16, 17, 17],
            [18, 19, 19],
            [20, 21, 21],
            [22, 23, 23],
            [24, 25, 25],
            [26, 27, 27],
            [28, 29, 29]
        ])

        mesh = gl.GLMeshItem(vertexes=verts, faces=faces, smooth=False,
                             drawFaces=False, drawEdges=True, edgeColor=(0.8, 0.8, 0.8, 1))
        view.addItem(mesh)

    def drawInnerMarks(self, view, stepOffset):
        zStart = self.corners[4] - stepOffset
        zEnd = self.corners[5] - stepOffset

        xmid = np.average(self.corners[0:2])
        ymid = np.average(self.corners[2:4])
        zmid = (zStart + zEnd) / 2

        (xmin, xmax) = self.corners[0:2]
        (ymin, ymax) = self.corners[2:4]
        (zmin, zmax) = (zStart, zEnd)

        #zmax = zmin + (zmax / 5)
        #zmid = (zmax - zmin) / 2

        verts = np.array([
            [xmin, ymid, zmin],
            [xmax, ymid, zmin],
            [xmid, ymin, zmin],
            [xmid, ymax, zmin],

            [xmin, ymid, zmax],
            [xmax, ymid, zmax],
            [xmid, ymin, zmax],
            [xmid, ymax, zmax],

            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmin, ymax, zmin],
            [xmax, ymax, zmin],

            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmin, ymax, zmax],
            [xmax, ymax, zmax],

            [xmin, ymin, zmin],
            [xmin, ymax, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],

            [xmin, ymin, zmax],
            [xmin, ymax, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
        ])
        faces = np.array([
            [0, 1, 1],
            [2, 3, 3],
            [4, 5, 5],
            [6, 7, 7],
            [8, 9, 9],
            [10, 11, 11],
            [12, 13, 13],
            [14, 15, 15],
            [16, 17, 17],
            [18, 19, 19],
            [20, 21, 21],
            [22, 23, 23]
        ])

        mesh = gl.GLMeshItem(vertexes=verts, faces=faces, smooth=False,
                             drawFaces=False, drawEdges=True, edgeColor=(1, 1, 1, 0.4))
        view.addItem(mesh)
