"""
@author: Cole Hill
University of South Florida
Department of Computer Science and Engineering
Computer Vision
Spring 2020
"""

import numpy as np


def sort_rect(points):
    """
    Accepts an array of 4 points and sorts them
    [lower_left, upper_left, lower_right, upper_right]
    """
    #Sort points so [Left|Right]
    points = points[0:4]
    x_sort = sorted(points,key=lambda x: x[0])

    left = x_sort[:2]
    right = x_sort[2:]

    #Sort left and right half so [Lower|Upper]
    left = sorted(left,key=lambda x: -x[1])
    right = sorted(right, key=lambda x : -x[1])
    return left+right

class Quadrilateral:
    """
    Class for defining a quadrilateral and establishing if points
    are withing the boundaries of the quadrilateral
    """
    def __init__(self,points):

        #Sort points
        tmp = sort_rect(points)
        #Convert points to numpy array
        self.points = []
        for pnt in tmp:
            self.points.append([*pnt])
        self.points = np.array(self.points)
        

    def in_rectangle(self, point):
        """
        Determines if `point` is located within the rectangle
        """

        y,x = point
        
        #Organize rectangle points into lines defining top,bottom,left,right
        top_line = self.points[[1,3],:]
        bottom_line = self.points[[0,2],:]
        left_line = self.points[[0,1],:]
        right_line = self.points[[2,3],:]

        #Interpolate and determine, top,bottom,left,right bounds for rectangle
        #at point x,y
        top = np.interp(x,top_line[:,1],top_line[:,0])
        bottom = np.interp(x,bottom_line[:,1],bottom_line[:,0])
        left = np.interp(y,left_line[:,0],left_line[:,1])
        right = np.interp(y,right_line[:,0],right_line[:,1])

        #If x,y tuple in bounds return true
        if left<=x<=right:
            if bottom<=y<=top:
                return True
        return False
        
    def get_x_bounds(self,y):
        """
        For a given line f(x) = y, determine
        the range on x axis such that f(x) is within
        the rectangle
        """
        left_line = self.points[[0,1],:]
        right_line = self.points[[2,3],:]

        left = np.interp(y,left_line[:,1],left_line[:,0])
        right = np.interp(y,right_line[:,1],right_line[:,0])
        
        return left,right
    
    def get_y_bounds(self,x):
        """
        For a given line f(y) = x, determine
        the range on y axis such that f(y) is within
        the rectangle
        """
        top_line = self.points[[1,3],:]
        bottom_line = self.points[[0,2],:]

        top = np.interp(x,top_line[:,0],top_line[:,1])
        bottom = np.interp(x,bottom_line[:,0],bottom_line[:,1])

        return bottom,top

if __name__ == "__main__":

    pass
    