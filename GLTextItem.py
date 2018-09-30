import pyqtgraph.opengl as gl
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from pyqtgraph.Qt import QtCore, QtGui

# Helper class for rendering text in pyqtgraphs 3D GLViewWidget
# Courtesy of Xinfa Zhu (https://groups.google.com/forum/#!topic/pyqtgraph/JHviWSaGhck)
# With modifications to the constructor and the color attribute
class GLTextItem(GLGraphicsItem):
    def __init__(self, GLViewWidget=None, X=None, Y=None, Z=None, text=None, color=QtCore.Qt.white):
        GLGraphicsItem.__init__(self)

        self.GLViewWidget = GLViewWidget
        self.text = text
        self.color = color
        self.X = X
        self.Y = Y
        self.Z = Z

    def setGLViewWidget(self, GLViewWidget):
        self.GLViewWidget = GLViewWidget

    def setText(self, text):
        self.text = text
        self.update()

    def setX(self, X):
        self.X = X
        self.update()

    def setY(self, Y):
        self.Y = Y
        self.update()

    def setZ(self, Z):
        self.Z = Z
        self.update()

    def paint(self):
        self.GLViewWidget.qglColor(self.color)
        self.GLViewWidget.renderText(self.X, self.Y, self.Z, self.text)