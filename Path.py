#Copyright 2018 Dennis Schunder
#
#Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
#
#http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT #WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy as np
import matplotlib.pyplot as plt

# Class for displaying and comparing the similarity of paths
class Path(object):

    def __init__(self, gui, index, pid, points, size=1, similarity=None, similarities=None, children=None, show=True):
        self.index = index
        self.pid = pid
        self.points = points
        self.size = size
        self.similarity = similarity
        self.children = children
        self.lines = None
        self.lineInfo = None
        self.visible = False

        self.validInds = self.points[2, self.points[2] != -1].astype(int)

        self.sizes = self.points[3]

        if similarities is None:
            self.similarities = np.zeros(len(self.validInds))
        else:
            self.similarities = similarities[self.validInds]


        if show:
            self.draw(gui)

    def __getstate__(self):
        save = {}

        for x in self.__dict__:
            if x == 'lines' or x == 'visible':
                continue
            save[x] = self.__dict__[x]
        return save

    def __setstate__(self, state):
        self.__dict__ = state
        self.lines = None
        self.lineInfo = None
        self.visible = False


    def getPoints(self):
        return self.points

    def getSize(self):
        return self.size

    def getColor(self, colorMode=0):
        duration = len(self.validInds)

        if colorMode == 0:
            r = np.linspace(0, 1, duration)
            g = [0] * duration
            b = np.linspace(0, 1, duration)[::-1]
            
        elif colorMode == 1:
            color = plt.get_cmap('viridis')

            (r,g,b) = self.map(self.similarities, color.colors)

        return np.column_stack((r,g,b))

    def getWidth(self, widthMode=0):
        duration = len(self.validInds)

        if widthMode == 0:
            return np.array([self.size] * duration)
        elif widthMode == 1:
            # Construct width based on size of grid section
            s = self.sizes - min(self.sizes)
            x = np.divide(s, max(s)) * 10
            return self.size + x

    def updateColor(self, color=None, colorMode=0):
        if self.lines != None:
            if color is None:
                color = self.getColor(colorMode=colorMode)

            index = 0

            for i in range(len(self.lines)):
                endIndex = index + self.lineInfo[i]
                self.lines[i].color = color[index:endIndex+1]
                index = endIndex

    def updateWidth(self, gui, widthMode, colorMode):
        if self.lines != None:
            self.hide(gui)
            self.draw(gui, colorMode, widthMode)

    def map(self, x, color):
        y = np.linspace(0, 1, len(color))
        c = np.empty((len(x), 3))
        
        for i in range(len(x)):
            for j in range(len(color)):
                if x[i] <= y[j]:
                    c[i] = color[j]
                    break

        return (c[:,0], c[:,1], c[:,2])


    def draw(self, gui, colorMode=0, widthMode=0, showChildren=False):
        if self.visible:
            return

        if len(self.validInds) == 0:
            return

        if not showChildren and self.children != None:
            for child in self.children:
                child.hide(gui, colorMode=colorMode, widthMode=widthMode)

        self.lines = []
        self.lineInfo = []

        colors = self.getColor(colorMode)
        widths = self.getWidth(widthMode)
        lastWidth = widths[0]
        startIndex = 0

        # Iterate over widths to create paths of same width
        for i in range(len(widths)):
            if widths[i] != lastWidth or i == len(widths) - 1:
                endIndex = i + 1

                linePoints = self.points[:3, startIndex:endIndex]
                lineColors = colors[startIndex:endIndex, :]

                self.lineInfo.append(endIndex-startIndex)
                self.lines.append(gui.drawPath(linePoints, lineColors, lastWidth))
                startIndex = i
                lastWidth = widths[i]
        self.visible = True

    def hide(self, gui, colorMode=0, widthMode=0, showChildren=False):
        if not self.visible:
            return

        if showChildren and self.children != None:
            for child in self.children:
                child.draw(gui, colorMode=colorMode, widthMode=widthMode)

        for line in self.lines:
            gui.hidePath(line)
        self.lines = None
        self.visible = False

    def merge(self, gui, pid, path, similarity, show=True):
        #Merge the two paths
        pathPoints = path.getPoints()

        pointAmt = len(pathPoints[2])

        newSize = 1
        multSelf = 1
        multOther = 1

        # Calculate relative weights
        if path.getSize() > self.size:
            multOther = self.size / path.size
            multSelf = 1 - multOther
            newSize = newSize + path.size
        else:
            multOther = path.size / self.size
            multSelf = 1 - multOther
            newSize = newSize + self.size

        newPoints = np.zeros((4, pointAmt))



        # Find valid entries for each path
        indsOther = path.validInds
        indsSelf = self.validInds
        
        # Get the steps present in both
        indsBoth = np.intersect1d(indsSelf, indsOther)

        # Find unique values in each path and set them
        indsSelfNew = np.setdiff1d(indsSelf, indsOther)
        indsOtherNew = np.setdiff1d(indsOther, indsSelf)

        newPoints[:2, indsSelfNew] = self.points[:2, indsSelfNew]
        newPoints[:2, indsOtherNew] = pathPoints[:2, indsOtherNew]

        newPoints[3, indsSelfNew] = self.points[3, indsSelfNew]
        newPoints[3, indsOtherNew] = pathPoints[3, indsOtherNew]

        #Average x, y inputs where both paths share the same timestep
        if len(indsBoth) != 0:

            # Extract x,y points and weigh them accordingly
            selfPoints = self.points[:2, indsBoth] * multSelf
            otherPoints = pathPoints[:2, indsBoth] * multOther

            newPoints[:2, indsBoth] = np.add(selfPoints, otherPoints)

            # Repeat for their group sizes
            selfSizes = self.points[3, indsBoth] * multSelf
            otherSizes = pathPoints[3, indsBoth] * multOther

            newPoints[3, indsBoth] = np.add(selfSizes, otherSizes)


            #newPoints[:2, indsBoth] = np.average((selfPoints, otherPoints), axis=0)


        newPoints[2] = np.full(pointAmt, -1)

        indsAll = np.union1d(indsSelf, indsOther)
        for i in indsAll:
            newPoints[2, i] = i


        # Calculate similarity between the individual points of the paths 
        diffs = np.subtract(self.points[:2], pathPoints[:2])
        squared = np.square(diffs)
        distances = np.sqrt(np.sum(squared, axis=0))
        distSim = np.divide(distances, np.max(distances))
        similarities = np.subtract(1, distSim)


        if show:
            path.hide(gui)
            self.hide(gui)

        return Path(gui, pid, pid, newPoints, newSize, similarity, similarities, (path, self), show)


    # Calculates the similarity of this path to the given one, under respect of the given maxDistance
    def calculateSimilarity(self, path, maxDistance):
        pathPoints = path.getPoints()

        inds = np.intersect1d(self.validInds, path.validInds)

        # The paths have no similarity
        if len(inds) == 0:
            return 0
        
        # Calculate similarity between the two paths
        diffs = np.subtract(self.points[:2, inds], pathPoints[:2, inds])
        squared = np.square(diffs)
        distances = np.sqrt(np.sum(squared, axis=0))
        distSim = np.divide(distances, maxDistance)
        similarities = np.subtract(1, distSim)

        return np.average(similarities)

