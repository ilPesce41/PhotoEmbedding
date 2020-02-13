from imageio import imread,imwrite
import numpy as np
import numba
import matplotlib.pyplot as plt
import matplotlib
from PyQt5.QtWidgets import QApplication,QMainWindow
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvas
import sys
import os

class ClickCanvas(FigureCanvas):

    def __init__(self,fig):

        super().__init__(fig)
        self.fig = fig
        self.ax = fig.subplots(1)
        self.points = []
        self.fig.canvas.mpl_connect('button_press_event',self.on_click)
        self.dot_artists = []

    
    def on_click(self,event):
        
        x,y = int(event.xdata),int(event.ydata)
        self.points.append((x,y))
        while len(self.points)>4:
            self.points = self.points[1:]
        for art in self.dot_artists:    
            art.remove()
        self.dot_artists = [] 
        for pnt in self.points:
            self.dot_artists.append(self.ax.scatter(pnt[0],pnt[1],color='r'))
        self.draw()


class ImageEmbedWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        widget = QtWidgets.QWidget()

        vbox = QtWidgets.QVBoxLayout()
        self.figcanvas = ClickCanvas(plt.Figure())
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

        self.background = None
        self.embedded = None
        self.modified = None

        save_image.clicked.connect(self.save_image)
        clear.clicked.connect(self.clear)
        embed_plots.clicked.connect(self.embed_image)


    def import_file(self,line_edit):

        fp,_ = QtWidgets.QFileDialog().getOpenFileName(parent=self,caption="Select File")
        if _:
            line_edit.setText(fp)


    def save_image(self):

        if not self.modified is None:

            fp,_ = QtWidgets.QFileDialog().getSaveFileName(parent = self, caption = "Save Image")
            
            if _:
                os.makedirs(os.path.split(fp)[0],exist_ok=True)
                imwrite(fp,self.modified)

    def plotBackground(self):

        try:
            self.background = imread(self.embed_file.text())
            self.figcanvas.ax.imshow(self.background)
            self.figcanvas.draw()
        except Exception as e:
            print(e)

    def clear(self):
        
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

        points = sorted(self.figcanvas.points)
        if len(points)!=4:
            return
        
        try:
            self.embedded = imread(self.embedded_file.text())
        except:
            return
        
        xlim,ylim = self.embedded.shape[0:2]
        npoints = sorted([(0,0),(xlim,0),(ylim,0),(xlim,ylim)])

        xi = []
        xip = []

        for i in range(len(npoints)):
            xi.append([npoints[i][0],npoints[i][1]])
            xip.append([points[i][0],points[i][1]])

        xi = np.array(xi)
        xip = np.array(xip)
        p = get_projection_params(xi,xip)

        self.modified = project_image(self.background,self.embedded,p)
        self.figcanvas.ax.clear()
        self.figcanvas.points = []
        self.figcanvas.dot_artists = []
        self.figcanvas.ax.imshow(self.modified)
        self.figcanvas.draw()
        


def build_hessian(x,y):
    """
    Takes array of x and y points in original image,
    returns hessian for affine projection
    """
    x_sum = np.sum(x)
    x_squared_sum = np.sum(x*x)
    y_sum = np.sum(y)
    y_squared_sum = np.sum(y*y)
    xy_sum = np.sum(x*y)

    hessian = np.array([
        [1,0,x_sum,y_sum,0,0],
        [0,1,0,0,x_sum,y_sum],
        [x_sum,0,x_squared_sum,xy_sum,0,0],
        [y_sum,0,xy_sum,y_squared_sum,0,0],
        [0,x_sum,0,0,x_squared_sum,xy_sum],
        [0,y_sum,0,0,xy_sum,y_squared_sum]
    ])

    return hessian

def build_b_mat(xi,xip):
    """
    Given arrays of original points and projected points,
    returns b matrix
    """
    
    xdif = xi[:,0] - xip[:,0]
    x,y = xi[:,0],xi[:,1]
    ydif = xi[:,1] - xip[:,1]

    vsum = np.sum

    b_mat = np.array([
        [vsum(xdif)],
        [vsum(ydif)],
        [vsum(x*xdif)],
        [vsum(y*xdif)],
        [vsum(x*ydif)],
        [vsum(y*ydif)]
    ])

    return b_mat


def get_projection_params(xi,xip):
    """
    Given a set of original points and projected points
    returns affine projection parameters as an array
    [tx,ty,a00,a01,a10,a11]

    *a00,a11 still need to be incrementeed by 1.0
    """
    A = build_hessian(xi[:,0],xi[:,1])
    b = build_b_mat(xi,xip)

    try:
        p= np.linalg.inv(A)@b
    except:
        p= np.linalg.pinv(A)@b
    p = p[:,0].T
    return p


def project_image(image,embedded,p):
    """
    Embedds and image `embedded` in `image` using
    affine projection with paramters `p`

    `p` = [tx,ty,a00,a01,a10,a11]
    """
    nimage = np.copy(image)
    
    tx,ty,a00,a01,a10,a11 = p
    print(tx,ty,a00,a01,a10,a11)
    H = np.array([
        [1+a00,a01,tx],
        [a10,1+a11,ty],
        [0,0,1]
    ])

    xlim,ylim = image.shape[0],image.shape[1]

    for x in range(embedded.shape[0]):
        for y in range(embedded.shape[1]):
            xi = np.array([x,y,1]).T
            xip = H@xi
            xp,yp = int(xip[0]),int(xip[1])
            if xp>0 and xp<xlim:
                if yp>0 and yp<ylim:
                    nimage[xp,yp] = embedded[x,y]
    
    return nimage
            

if __name__ == "__main__":

    Application = QApplication(sys.argv)

    window = ImageEmbedWindow()
    window.show()

    Application.exec_()



