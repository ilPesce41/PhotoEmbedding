import numpy as np



def sort_rect(points):
    """
    Accepts an array of 4 points and sorts them
    [lower_left, upper_left, lower_right, upper_right]
    """
    points = points[0:4]
    x_sort = sorted(points,key=lambda x: x[0])

    left = x_sort[:2]
    right = x_sort[2:]

    left = sorted(left,key=lambda x: -x[1])
    right = sorted(right, key=lambda x : -x[1])
    print(left+right)
    return left+right

class Rectangle:

    def __init__(self,points):

        tmp = sort_rect(points)
        self.points = []
        for pnt in tmp:
            self.points.append([*pnt])
        self.points = np.array(self.points)
        

    def in_rectangle(self, point):

        y,x = point

        top_line = self.points[[1,3],:]
        bottom_line = self.points[[0,2],:]
        left_line = self.points[[0,1],:]
        right_line = self.points[[2,3],:]

        top = np.interp(x,top_line[:,1],top_line[:,0])
        bottom = np.interp(x,bottom_line[:,1],bottom_line[:,0])
        left = np.interp(y,left_line[:,0],left_line[:,1])
        right = np.interp(y,right_line[:,0],right_line[:,1])

        if left<=x<=right:
            if bottom<=y<=top:
                return True
        return False
        
    def get_x_bounds(self,y):

        left_line = self.points[[0,1],:]
        right_line = self.points[[2,3],:]

        left = np.interp(y,left_line[:,1],left_line[:,0])
        right = np.interp(y,right_line[:,1],right_line[:,0])
        
        return left,right
    
    def get_y_bounds(self,x):

        top_line = self.points[[1,3],:]
        bottom_line = self.points[[0,2],:]

        top = np.interp(x,top_line[:,0],top_line[:,1])
        bottom = np.interp(x,bottom_line[:,0],bottom_line[:,1])

        return bottom,top

if __name__ == "__main__":

    points = [
        (0,0),
        (1,1.1),
        (1,0),
        (0,1)
    ]

    rect = Rectangle(points)
    print(rect.in_rectangle((2,0)))
    print(rect.in_rectangle((.1,1.5)))
    