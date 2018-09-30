#Copyright 2018 Dennis Schunder
#
#Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
#
#http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT #WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

from PyQt5 import QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import GLTextItem as glt
import numpy as np
from Ui_MainWindow import Ui_MainWindow
import matplotlib.pyplot as plt
import sys
from OctTree import OctTree
from Path import Path
import h5py
from timeit import default_timer as timer
import jsonpickle
import jsonpickle.ext.numpy
import os.path
import ntpath


class GUI:

    app = None
    mw = None
    pw = None

    currentScale = [1, 1, 1]
    
    dimensionString = "[%.2f|%.2f|%d] - [%.2f|%.2f|%d]"
    positionString = "[%.2f|%.2f|%d]"

    sizeString = "[%d - %d]"
    simString = "[%.4f - %.4f]"

    #Sets up the GUI
    def __init__(self):
        self.app = QtGui.QApplication(sys.argv)

        jsonpickle.ext.numpy.register_handlers()

        mw = QtGui.QMainWindow()
        ui = Ui_MainWindow()
        ui.setupUi(mw)

        #Obtain DesktopWidget to gain screen sizes
        dw = QtGui.QDesktopWidget()
        screen = dw.availableGeometry()

        #Resize main window to 90% of the screen width/height
        mw.resize(screen.size() * 0.9)

        #Recenter main window
        frame = mw.frameGeometry()
        frame.moveCenter(screen.center())
        mw.move(frame.topLeft())

        pw = gl.GLViewWidget(ui.centralwidget)
        ui.gridLayout.replaceWidget(ui.plotWidget, pw)
        ui.plotWidget.hide()
        ui.plotWidget = pw

        self.ui = ui
        self.mw = mw
        self.pw = pw

        self.tree = None
        self.paths = None
        self.seedAmt = None
        self.state = -1

        mw.setWindowTitle("Hierarchical Visualizer")

        pw.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        axis = gl.GLAxisItem()
        xText = glt.GLTextItem(GLViewWidget=self.pw, X=1, Y=0, Z=0, text="X")
        yText = glt.GLTextItem(GLViewWidget=self.pw, X=0, Y=1, Z=0, text="Y")
        zText = glt.GLTextItem(GLViewWidget=self.pw, X=0, Y=0, Z=1, text="Z")

        self.pw.addItem(axis)
        self.pw.addItem(xText)
        self.pw.addItem(yText)
        self.pw.addItem(zText)

        self.connectSlots()

    def connectSlots(self):

        # File open dialog
        self.ui.actionOpen.triggered.connect(self.openFile)

        #Data load dialog
        self.ui.actionLoad_data.triggered.connect(self.loadFile)

        #Data save dialog
        self.ui.actionSave_as.triggered.connect(self.saveFile)

        # Tree build button
        self.ui.BuildTreeAction.clicked.connect(self.buildTree)

        #Path Buttons
        self.ui.PlacePaths.clicked.connect(self.placePaths)
        self.ui.CreateHierarchy.clicked.connect(self.createHierarchy)

        # Area meaning selection
        self.ui.AreaValue.currentIndexChanged.connect(self.areaChanged)

        # Color meaning selection
        self.ui.ColorValue.currentIndexChanged.connect(self.colorChanged)
        

        # Connect scale inputs to the scaling functions
        self.ui.xScale.valueChanged.connect(self.updateXNum)
        self.ui.xNum.valueChanged.connect(self.updateXSlider)
        self.ui.xNum.valueChanged.connect(self.xScale)

        self.ui.yScale.valueChanged.connect(self.updateYNum)
        self.ui.yNum.valueChanged.connect(self.updateYSlider)
        self.ui.yNum.valueChanged.connect(self.yScale)

        self.ui.zScale.valueChanged.connect(self.updateZNum)
        self.ui.zNum.valueChanged.connect(self.updateZSlider)
        self.ui.zNum.valueChanged.connect(self.zScale)

        #Connect Hierarchy slider functions
        self.ui.HierDepthSlider.valueChanged.connect(self.changeHierDepth)
        self.ui.HierDepthNum.valueChanged.connect(self.updateHierSlider)

        #Connect the highlight toggle options
        self.ui.actionLast_Children.toggled.connect(self.highlightNextChanged)
        self.ui.actionLast_Merge.toggled.connect(self.highlightCurrChanged)

        #Connect the wireframe toggle options
        self.ui.actionShow_Wireframe.toggled.connect(self.wireframeChange)
        self.ui.actionShow_Timeslices.toggled.connect(self.wireframeChange)
        self.ui.actionShow_Inner_Bounds.toggled.connect(self.wireframeChange)
        self.ui.actionShow_z_Lines.toggled.connect(self.wireframeChange)

    def wireframeChange(self, newVal):
        items = self.pw.items.copy()

        for c in items:
            if isinstance(c, (gl.GLBoxItem, gl.GLMeshItem)):
                self.pw.removeItem(c)

        if self.tree is None:
            return

        self.tree.draw(self.pw, self.dimensions[2], self.ui.actionShow_Wireframe.isChecked(), self.ui.actionShow_Timeslices.isChecked(), self.ui.actionShow_Inner_Bounds.isChecked(), self.ui.actionShow_z_Lines.isChecked())
        
        for c in self.pw.items:
            if isinstance(c, (gl.GLBoxItem, gl.GLMeshItem)):
                c.scale(self.currentScale[0], self.currentScale[1], self.currentScale[2], False)

    # Handle the change of the highlighting of the current merge
    def highlightCurrChanged(self, newVal):
        self.highlightMerge = newVal

        if not self.ui.VisControl.isEnabled():
            return

        currIndex = self.ui.HierDepthNum.value() + self.seedAmt - 1

        if not newVal:
            self.unhighlight(currIndex)
        else:
            self.highlight(currIndex, self.mergeColor)

    # Handle the change of the highlighting next merge children option
    def highlightNextChanged(self, newVal):
        self.highlightNextMerge = newVal

        if not self.ui.VisControl.isEnabled():
            return
        
        for c in self.lastChildHighlight:
            # If the current merge is highlighted, do not change its color
            if self.highlightMerge:
                currIndex = self.ui.HierDepthNum.value() + self.seedAmt - 1
                if c.index == currIndex:
                    continue

            if not newVal:
                self.unhighlight(c.index)
            else:
                self.highlight(c.index, self.nextColor)


    def areaChanged(self, index):
        for path in self.paths:
            path.updateWidth(self, index, self.ui.ColorValue.currentIndex())

    def colorChanged(self, index):
        for path in self.paths:
            path.updateColor(colorMode=index)
        self.pw.update()

    def openFile(self):
        dlg = QtGui.QFileDialog(self.mw)
        title = "Open Dataset..."
        h5filter = "HDF5 files (*.h5part *.hdf5)"
        
        f = QtGui.QFileDialog.getOpenFileName(dlg, title, filter=h5filter)

        try:
            self.file = h5py.File(f[0], 'r')
            self.fileName = ntpath.basename(f[0])

            self.tree = None
            self.paths = None
            self.seedAmt = None

            items = self.pw.items.copy()

            for child in items:
                if isinstance(child, (gl.GLLinePlotItem, gl.GLBoxItem, gl.GLMeshItem)):
                    self.pw.removeItem(child)

            self.mw.setWindowTitle(f[0])
            self.ui.TreeControl.setEnabled(True)
            self.ui.BuildTreeAction.setEnabled(True)
            self.ui.VisControl.setEnabled(False)
            self.ui.PlotControl.setEnabled(False)
            self.ui.currentScale = [1,1,1]
            self.ui.xScale.setValue(1)
            self.ui.yScale.setValue(1)
            self.ui.zScale.setValue(1)
            self.ui.xNum.setValue(1)
            self.ui.yNum.setValue(1)
            self.ui.zNum.setValue(1)
            self.ui.PathProgress.setValue(0)
            self.ui.TreeProgress.setValue(0)
            self.ui.StatusValue.setText("File loaded: %s" % self.fileName)
            self.ui.HierDepthNum.setValue(0)
            self.ui.HierDepthSlider.setValue(0)
            self.state = 0

        except ValueError:
            return

    def saveFile(self):
        if self.state == -1:
            return

        dlg = QtGui.QFileDialog(self.mw)
        title = "Save to..."
        
        f = QtGui.QFileDialog.getExistingDirectory(dlg, title)

        if self.save(dir=f):
            self.ui.SavedStateVal.setText("True")


    def treeDone(self):

        self.dimensions = list(self.tree.corners[0::2])
        self.dimensions = self.dimensions + list(self.tree.corners[1::2])
        self.dimensions = tuple(self.dimensions)

        self.ui.DimensionsValue.setText(self.dimensionString % self.dimensions)

        
        self.ui.StatusValue.setText("Tree built")
        self.ui.PlotControl.setEnabled(True)
        self.ui.VisControl.setEnabled(True)

        self.ui.BuildTreeAction.setEnabled(True)
        self.ui.PlacePaths.setEnabled(True)
        self.ui.CreateHierarchy.setEnabled(False)
        self.ui.HierDepthText.setEnabled(False)
        self.ui.HierDepthSlider.setEnabled(False)
        self.ui.HierDepthNum.setEnabled(False)
        self.ui.AreaValue.setEnabled(False)
        self.ui.ColorValue.setEnabled(False)

        self.ui.SimValue.setText(self.simString % self.tree.findSimilarityRange())
        self.ui.SizeValue.setText(self.sizeString % self.tree.findSizeRange())
        self.ui.LeafCountValue.setText("%d" % self.tree.countLeaves())

        self.state = 1

    def pathsDone(self):
        self.ui.StatusValue.setText("Paths seeded")
        self.ui.CreateHierarchy.setEnabled(True)
        self.ui.PlacePaths.setEnabled(False)
        self.ui.PathProgress.setValue(self.ui.PathProgress.maximum())

        print("Starting Paths seeded")
        self.state = 2
    
    def hierDone(self):

        print("Hierarchy created")

        self.ui.StatusValue.setText("Hierarchy created")

        self.lastChildHighlight = None
        
        # Highlighting
        pathLength = self.dimensions[5] - self.dimensions[2] + 1
        mergeColor = np.empty((pathLength, 3))
        mergeColor[:, 0] = [1.0] * pathLength
        mergeColor[:, 1] = [0.8] * pathLength
        mergeColor[:, 2] = [0.0] * pathLength

        nextColor = np.empty((pathLength, 3))
        nextColor[:, 0] = [0.0] * pathLength
        nextColor[:, 1] = [0.4] * pathLength
        nextColor[:, 2] = [0.0] * pathLength

        self.mergeColor = mergeColor
        self.nextColor = nextColor

        self.highlightMerge = True
        self.highlightNextMerge = True

        self.hierDepth = 0

        self.ui.HierDepthSlider.setMaximum(self.mergeAmt)
        self.ui.HierDepthNum.setMaximum(self.mergeAmt)

        self.ui.HierDepthText.setEnabled(True)
        self.ui.HierDepthSlider.setEnabled(True)
        self.ui.HierDepthNum.setEnabled(True)

        self.ui.AreaValue.setEnabled(True)
        self.ui.ColorValue.setEnabled(True)

        self.ui.PathProgress.setValue(self.ui.PathProgress.maximum())
        self.ui.HierDepthVal.setText("%d" % self.ui.HierDepthNum.maximum())

        self.ui.CreateHierarchy.setEnabled(False)
        self.state = 3

        self.applyHighlighting(self.seedAmt, -1)

    def loadFile(self):
        if self.state == -1:
            return

        dlg = QtGui.QFileDialog(self.mw)
        title = "Open from..."
        
        f = QtGui.QFileDialog.getExistingDirectory(dlg, title)

        if self.load(dir=f):
            self.ui.SavedStateVal.setText("True")

    def load(self, dir=''):
        print("Detected pickled data, unpickling...")

        # Load the tree from the pickled json string
        pickleFile = open(dir + self.pickleID, 'r')
        jsonString = pickleFile.read()
        pickleFile.close()

        saved = jsonpickle.decode(jsonString)

        self.tree = saved['tree']
        self.paths = np.array(saved['paths'])
        self.seedAmt = saved['seeds']
        self.state = saved['state']

        self.mergeAmt = len(self.paths) - self.seedAmt 

        startState = self.state

        if startState > 0:
            self.treeDone()
        if startState > 1:
            for path in self.paths[:self.seedAmt]:
                path.draw(self, mode=self.ui.ColorValue.currentIndex())
            self.pathsDone()
        if startState > 2:
            self.hierDone()

        print("Restored data with state %d" % startState)

    def save(self, dir=''):
        print("Pickling...")

        tree = self.tree
        paths = self.paths
        seeds = self.seedAmt

        if not paths is None:
            paths = paths.tolist()

        # Write important data into file
        save = {'tree': tree, 'paths': paths, 'seeds': seeds, 'state': self.state}

        # Encode into json
        jsonString = jsonpickle.encode(save)

        try:

            # Write into pickle file
            pickleFile = open(dir + '/' + self.pickleID, 'w')
            pickleFile.write(jsonString)
            pickleFile.close()
        
        except ValueError:
            print("Error ocurred while pickling")
            return False

        print("Done pickling...")
        return True


    def buildTree(self):
        f = self.file

        items = self.pw.items.copy()

        # Delete any previous items
        for child in items:
                if isinstance(child, (gl.GLLinePlotItem, gl.GLBoxItem, gl.GLMeshItem)):
                    self.pw.removeItem(child)

        # Reset scaling
        self.ui.currentScale = [1,1,1]
        self.ui.xScale.setValue(1)
        self.ui.yScale.setValue(1)
        self.ui.zScale.setValue(1)
        self.ui.xNum.setValue(1)
        self.ui.yNum.setValue(1)
        self.ui.zNum.setValue(1)

        groups = [f[group] for group in f]

        # List of all points in the file
        allPoints = []

        prg = 0
        maxPrg = len(groups) * 4
        self.ui.TreeProgress.setMaximum(maxPrg)

        for group in groups:
            step = int(group.name.split('#')[1])

            # Extract points of step
            try:
                points = (step, (np.copy(group['id']), (np.copy(group['X']), np.copy(group['Y'])), (np.copy(group['U']), np.copy(group['V']))))
            except KeyError:
                # Fix for lowercase velocity index
                points = (step, (np.copy(group['id']), (np.copy(group['X']), np.copy(group['Y'])), (np.copy(group['u']), np.copy(group['v']))))

            # Insert at the right spot so that allPoints is sorted by steps
            i = 0
            for point in allPoints:
                if step < point[0]:
                    break
                i = i + 1
            allPoints.insert(i, points)

            prg = prg + 1
            self.ui.TreeProgress.setValue(prg)


        minDepth = self.ui.MinDepthValue.value()
        maxDepth = self.ui.MaxDepthValue.value()
        eps = self.ui.EpsilonValue.value()

        # Instantiate tree
        self.tree = OctTree(allPoints, (minDepth, maxDepth), eps)
        self.tree.draw(self.pw, self.tree.getStartStep(), self.ui.actionShow_Wireframe.isChecked(), self.ui.actionShow_Timeslices.isChecked(), self.ui.actionShow_Inner_Bounds.isChecked(), self.ui.actionShow_z_Lines.isChecked())
        self.ui.TreeProgress.setValue(maxPrg)

        self.treeDone()

        self.pickleID = '[%s]-[%d-%d %.2f%%].json' % (self.fileName, minDepth, maxDepth, (eps * 100))

        if os.path.isfile(self.pickleID):
            self.load()


    def highlight(self, pid, color):
        self.paths[pid].updateColor(color=color)
        self.pw.update()

    def unhighlight(self, pid):
        self.paths[pid].updateColor(colorMode=self.ui.ColorValue.currentIndex())
        self.pw.update()

    def applyHighlighting(self, newIndex, oldIndex):
        prevInd = newIndex - 1

        # Highlight next merged paths
        if self.highlightNextMerge:
            if self.lastChildHighlight != None:
                for c in self.lastChildHighlight:
                    self.unhighlight(c.index)

            if newIndex < len(self.paths):
                self.lastChildHighlight = self.paths[newIndex].children

                for c in self.lastChildHighlight:
                    self.highlight(c.index, self.nextColor)

        # Highlight last merge
        if self.highlightMerge:
            if prevInd >= self.seedAmt:
                self.highlight(prevInd, self.mergeColor)
            self.unhighlight(oldIndex-1)


    def placePaths(self):
        ids = np.unique(self.tree.getVector())
        
        seedAmt = len(ids)

        pathAmt = 2 * seedAmt - 1

        print("%d seedpoints, resulting in %d paths" % (seedAmt, pathAmt))

        print("Seeding paths")

        paths = np.empty(pathAmt, dtype=object)

        self.ui.PathProgress.setMaximum(seedAmt)

        for i in range(seedAmt):
            path = self.tree.getPath(ids[i])

            paths[i] = Path(self, i, ids[i], path)
            self.ui.PathProgress.setValue(i + 1)
            print("%d / %d" % (i, seedAmt))

        self.paths = paths
        self.seedAmt = seedAmt
        self.pathAmt = pathAmt

        self.pathsDone()


    def createHierarchy(self):

        bounds = self.tree.corners

        maxDistance = np.sqrt(np.sum(np.square(np.subtract(bounds[1::2], bounds[0::2]))))

        print("Maximum distance is ", maxDistance)

        self.ui.PathProgress.setValue(0)

        print("Creating Similarity array...")

        start = timer()

        # Calculate amount of similarity checks using the sum of natural numbers
        simAmt = int((self.seedAmt * (self.seedAmt - 1)) / 2)
        
        # Amount of merges that can be done
        mergeAmt = self.pathAmt - self.seedAmt

        mergeSize = simAmt * 2 # Show merging as taking twice as long as calculating starting similarity
        mergeStepping = mergeSize / mergeAmt
        self.ui.PathProgress.setMaximum(mergeSize + simAmt)


        # Construct arrays for the pairs of path and their similarities
        pathPairs = np.empty((2, simAmt), dtype='i8')
        similarities = np.empty(simAmt)

        # Initialize similarities
        b = 0
        for i in range(0, self.seedAmt):
            for j in range(i + 1, self.seedAmt):
                # Set the specified index
                pathPairs[:, b] = (i, j)
                similarities[b] = self.paths[i].calculateSimilarity(self.paths[j], maxDistance)

                b = b + 1
                self.ui.PathProgress.setValue(b)

                
        end = timer()
        print("Took %fs" % (end - start))

        s = timer()

        # Helper variables to count 'deleted' elements
        skip = 0
        deleted = 0

        # Last created path
        lastPath = None

        # Merge paths until only one is left
        for i in range(mergeAmt):
            print("   %d/%d" % (i, mergeAmt))
            self.ui.PathProgress.setValue(self.ui.PathProgress.value() + mergeStepping)

            
            start = timer()

            # Find sorted indices, skipping 'deleted' elements
            sortInds = np.argsort(similarities[skip:])

            # Extract elements in sorted order and replace 'undeleted' part of array
            pathPairs[:, skip:] = pathPairs[:, skip:][:, sortInds]
            similarities[skip:] = similarities[skip:][sortInds]

            end = timer()
            print("      Sorting took %fs" % (end-start))

            #Calculate how many slots have been 'deleted'
            skip = skip + deleted
            deleted = 0

            shift = -1

            if lastPath != None:
                #Last element is latest created path
                shift = -2

            #Select tuple of highest similarity        
            (left, right) = pathPairs[:, shift]

            print("      Merging %d and %d" % (left, right))

            if left == -1:
                break

            #Merge selected paths
            newId = self.seedAmt + i
            start = timer()

            self.paths[newId] = self.paths[left].merge(self, newId, self.paths[right], similarities[shift], show=False)
            end = timer()
            print("      Merging took %fs" % (end - start))
            
            #Make sure to calculate similarity to last path if it wasn't merged
            if lastPath != None:
                if left != lastPath and right != lastPath:
                    pathPairs[:, shift] = (newId, lastPath)
                    similarities[shift] = self.paths[newId].calculateSimilarity(self.paths[lastPath], maxDistance)
                else:
                    pathPairs[:, shift] = (-1, -1)
                    similarities[shift] = -1
                    deleted = deleted + 1

            #Mark spot done
            pathPairs[:, -1] = (newId, newId)
            similarities[-1] = 1

            start = timer()
            #Fix invalid tuples
            for j in range(skip, simAmt):
                x = pathPairs[:, j]

                for k in range(2):
                    y = x[k]

                    #Delete similarity if right path
                    if y == right:
                        pathPairs[:, j] = (-1, -1)
                        similarities[j] = -1
                        deleted = deleted + 1
                        break

                    #Refresh similarity to new path if left path
                    if y == left:
                        otherPath = x[(k + 1) % 2]
                        
                        if otherPath != right:
                            pathPairs[:, j] = (newId, otherPath)
                            similarities[j] = self.paths[newId].calculateSimilarity(self.paths[otherPath], maxDistance)
                        else:
                            pathPairs[:, j] = (-1, -1)
                            similarities[j] = -1
                            deleted = deleted + 1
                        break                    

            end = timer()
            print("      Fixing took %fs" % (end - start))

            lastPath = newId

        e = timer()
        print("Took %fs" % (e - s))
        self.mergeAmt = mergeAmt
        self.hierDone()
        #self.save()

    
    def updateHierSlider(self, value):
        self.ui.HierDepthSlider.setValue(value)
        self.changeHierDepth(value)


    def changeHierDepth(self, newDepth):
        if newDepth == self.hierDepth or self.seedAmt is None:
            return

        oldIndex = self.hierDepth + self.seedAmt
        newIndex = newDepth + self.seedAmt

        if oldIndex > newIndex:
            # Splitting up all paths up to new index, show their children when hiding
            for i in reversed(range(newIndex, oldIndex)):
                self.paths[i].hide(self, colorMode=self.ui.ColorValue.currentIndex(), widthMode=self.ui.AreaValue.currentIndex(), showChildren=True)
        else:
            # Merging up all paths up to new index, hide children when drawing
            for i in range(oldIndex, newIndex):
                self.paths[i].draw(self, colorMode=self.ui.ColorValue.currentIndex(), widthMode=self.ui.AreaValue.currentIndex(), showChildren=False)

        self.applyHighlighting(newIndex, oldIndex)

        if newDepth == 0:
            self.paths[newDepth + self.seedAmt].hide(self, colorMode=self.ui.ColorValue.currentIndex(), widthMode=self.ui.AreaValue.currentIndex(), showChildren=True)

        self.hierDepth = newDepth


    # Base update function for scaling
    def updateNum(self, num, value):
        num.setValue((value / 100))
        return num.value()
    
    def updateSlider(self, num, slider):
        value = num.value()

        slider.setValue(int(value*100))

    def scale(self, scaling, relativeScaling=True):
        newScale = np.empty(3)

        if relativeScaling:
            for i in range(3):
                # Change the absolute scaling to a relative one
                if scaling[i] == -1:
                    newScale[i] = 1
                else:
                    newScale[i] = scaling[i] / self.currentScale[i]  # Calculate relative scaling
                    self.currentScale[i] = scaling[i]  # Save current absolute scale
        else:
            newScale = scaling


        for child in self.pw.items:
            child.scale(newScale[0], newScale[1], newScale[2], False)


    # Slots for scaling
    def updateXSlider(self, value):
        self.updateSlider(self.ui.xNum, self.ui.xScale)

    def updateYSlider(self, value):
        self.updateSlider(self.ui.yNum, self.ui.yScale)

    def updateZSlider(self, value):
        self.updateSlider(self.ui.zNum, self.ui.zScale)


    def updateXNum(self, value):
        self.scale((self.updateNum(self.ui.xNum, value), -1, -1))
        
    def updateYNum(self, value):
        self.scale((-1, self.updateNum(self.ui.yNum, value), -1))

    def updateZNum(self, value):
        self.scale((-1, -1, self.updateNum(self.ui.zNum, value)))
    

    def xScale(self, value):
        self.scale((self.ui.xNum.value(), -1, -1))

    def yScale(self, value):
        self.scale((-1, self.ui.yNum.value(), -1))

    def zScale(self, value):
        self.scale((-1, -1, self.ui.zNum.value()))





    def drawPath(self, path, color, size):

        path = path[:, path[2] != -1]

        newPath = np.column_stack(path)

        line = gl.GLLinePlotItem(pos=newPath, color=color, width=size, antialias=False)
        self.pw.addItem(line)
        line.scale(self.currentScale[0], self.currentScale[1], self.currentScale[2], False)
        return line

    def hidePath(self, line):
        self.pw.removeItem(line)

    def getDisplayWidget(self):
        return self.pw

    def show(self):
        self.mw.show()

        return self.app