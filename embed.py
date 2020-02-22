"""
@author: Cole Hill
University of South Florida
Department of Computer Science and Engineering
Computer Vision
Spring 2020
"""

#External Dependencies
from imageio import imread,imwrite
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PyQt5.QtWidgets import QApplication,QMainWindow
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvas
import sys
import os


from polygon import sort_rect,Quadrilateral
from projections import Affine,Homogeneous



class ClickCanvas(FigureCanvas):
    """
    Extension of matplotlib figure canvas to place marker on image
    when a user clicks. Tracks the location of the markers.
    4 marker maximum
    """
    def __init__(self,fig):

        super().__init__(fig)
        self.fig = fig
        #Initialize figure with single axis
        self.ax = fig.subplots(1)

        #Arrays to keep track of points/point artist
        self.points = []
        self.dot_artists = []

        #Hook up click event with marker placement fcn
        self.fig.canvas.mpl_connect('button_press_event',self.on_click)
        
    
    def on_click(self,event):
        """
        Handles when user clicks on image
        Places point on image using scatter and adds point
        to self.points list
        """
        
        #Get marker location
        x,y = int(event.xdata),int(event.ydata)
        self.points.append((x,y))
        
        #Limit to 4 points
        while len(self.points)>4:
            self.points = self.points[1:]
        
        #Remove all markers and replace
        for art in self.dot_artists:    
            art.remove()
        self.dot_artists = [] 
        for pnt in self.points:
            self.dot_artists.append(self.ax.scatter(pnt[0],pnt[1],color='r'))
        self.draw()


class ImageEmbedWindow(QMainWindow):
    """
    Class for importing two images `background` and `embedded`.
    Projects  `embedded` into `background` based on points in `background`
    selected by user
    """
    def __init__(self):

        super().__init__()

        #Figure canvas for plotting images and getting marker points
        self.figcanvas = ClickCanvas(plt.Figure())
        
        #Setup UI
        widget = QtWidgets.QWidget()

        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.figcanvas)
        
        hb1 = QtWidgets.QHBoxLayout()
        self.embed_file = QtWidgets.QLineEdit()
        self.embed_file.textChanged.connect(self.plotBackground)
        import_embed = QtWidgets.QPushButton("Import Background File")
        import_embed.clicked.connect(lambda x: self.import_file(self.embed_file))
        hb1.addWidget(self.embed_file)
        hb1.addWidget(import_embed)
        vbox.addLayout(hb1)

        hb2 = QtWidgets.QHBoxLayout()
        self.embedded_file = QtWidgets.QLineEdit()
        import_embedded = QtWidgets.QPushButton("Import Embedded File")
        import_embedded.clicked.connect(lambda x: self.import_file(self.embedded_file))
        hb2.addWidget(self.embedded_file)
        hb2.addWidget(import_embedded)
        vbox.addLayout(hb2)

        hb3 = QtWidgets.QHBoxLayout()
        embed_plots = QtWidgets.QPushButton("Embed Image")
        save_image = QtWidgets.QPushButton("Save Image")
        clear = QtWidgets.QPushButton("Clear Image")
        hb3.addWidget(embed_plots)
        hb3.addWidget(save_image)
        hb3.addWidget(clear)
        vbox.addLayout(hb3)

        widget.setLayout(vbox)
        self.setCentralWidget(widget)

        #3 image variables
        self.background = None
        self.embedded = None
        self.modified = None

        #hookup button events with callbacks
        save_image.clicked.connect(self.save_image)
        clear.clicked.connect(self.clear)
        embed_plots.clicked.connect(self.embed_image)


    def import_file(self,line_edit):
        """
        Opens dialog to query file path
        params:
        line_edit - line edit to populate with filepath
        """
        fp,_ = QtWidgets.QFileDialog().getOpenFileName(parent=self,caption="Select File")
        if _:
            line_edit.setText(fp)


    def save_image(self):
        """
        Opens dialog to query for file path to save modified image,
        saves image at specified filepath
        """
        if not self.modified is None:

            fp,_ = QtWidgets.QFileDialog().getSaveFileName(parent = self, caption = "Save Image")
            
            if _:
                os.makedirs(os.path.split(fp)[0],exist_ok=True)
                imwrite(fp,self.modified)

    def plotBackground(self):
        """
        Plots background image
        """
        self.clear()
        try:
            self.background = imread(self.embed_file.text())
            self.figcanvas.ax.imshow(self.background)
            self.figcanvas.draw()
        except Exception as e:
            print(e)

    def clear(self):
        """
        Clears out image and replots the background
        """
        self.figcanvas.ax.clear()
        self.figcanvas.fig.clear()
        self.figcanvas.ax = self.figcanvas.fig.subplots(1)
        self.figcanvas.points = []
        self.figcanvas.dot_artists =[]
        self.modified = None
        if not self.background is None:
            self.figcanvas.ax.imshow(self.background)
        self.figcanvas.draw()

    def embed_image(self):
        """
        Embeds `embedded` in `background` based on user points
        """

        #Get points from figure canvas
        points = sort_rect(self.figcanvas.points)
        #verify we have 4 points
        if len(points)!=4:
            return
        
        #Verify we have an image to embed
        try:
            self.embedded = imread(self.embedded_file.text())
        except:
            return
        
        #Establish corner points of image
        xlim,ylim = self.embedded.shape[0:2]
        npoints = sort_rect([(0,0),(ylim,0),(0,xlim),(ylim,xlim)])

        #Put points in to numpy arrays
        xi = []
        xip = []
        for i in range(len(npoints)):
            xi.append([npoints[i][0],npoints[i][1]])
            xip.append([points[i][0],points[i][1]])
        xi = np.array(xi)
        xip = np.array(xip)
        
        #Affine projection
        # affine = Affine(xip,xi)
        # H = affine.H
        #Homogeneous Projection
        projection = Homogeneous(xip,xi)
        H = projection.H

        #Project the image
        self.modified = project_image(self.background,self.embedded,H,Quadrilateral(points))
        
        #Plot projection
        self.figcanvas.ax.clear()
        self.figcanvas.points = []
        self.figcanvas.dot_artists = []
        self.figcanvas.ax.imshow(self.modified)
        self.figcanvas.draw()
        

def project_image(image,embedded,H,rect):
    """
    Embedds and image `embedded` in `image` using
    affine projection with paramters `H`

    """
    #Copy of background image to manipulate
    nimage = np.copy(image)
    
    #Establish image dimensions
    xs,ys = embedded.shape[1],embedded.shape[0]
    xlim,ylim = embedded.shape[1],embedded.shape[0]

    #Smallest necessary rectangle of points to scan
    xmin,ymin = np.min(rect.points,axis=0)
    xmax,ymax = np.max(rect.points,axis=0)
    
    #Scan through all xcords in rectangle
    for x in range(xmin,xmax+1):
        #Scan through all ycords for given xcord
        ymax,ymin = map(int,rect.get_y_bounds(x))
        for y in range(ymin,ymax+1):
                #Back project pixel location
                xi = np.array([x,y,1]).T
                xip = H@xi
                xp,yp,c = xip[0],xip[1],xip[2]
                xp = int(xp/c)
                yp = int(yp/c)
                #Ensure we are still in image bounds
                xp = np.max([0,xp])
                xp = np.min([xp,xlim-1])
                yp = np.max([0,yp])
                yp = np.min([yp,ylim-1])
                #Update pixel value
                nimage[y,x] = embedded[yp,xp]      
    return nimage
            

if __name__ == "__main__":

    Application = QApplication(sys.argv)

    window = ImageEmbedWindow()
    window.show()

    Application.exec_()
