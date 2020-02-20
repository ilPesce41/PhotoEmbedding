import numpy as np
class Affine:

    def __init__(self,points,projected):

        self.points = points
        self.projected = projected
        self.p = self.get_projection_params(points,projected)

    def build_hessian(self,x,y):
        """
        Takes array of x and y points in original image,
        returns hessian for affine projection
        """
        x_sum = np.sum(x)
        x_squared_sum = np.sum(x*x)
        y_sum = np.sum(y)
        y_squared_sum = np.sum(y*y)
        xy_sum = np.sum(x*y)
        n = len(x)
        hessian = np.array([
            [n,0,x_sum,y_sum,0,0],
            [0,n,0,0,x_sum,y_sum],
            [x_sum,0,x_squared_sum,xy_sum,0,0],
            [y_sum,0,xy_sum,y_squared_sum,0,0],
            [0,x_sum,0,0,x_squared_sum,xy_sum],
            [0,y_sum,0,0,xy_sum,y_squared_sum]
        ])
        hessian = hessian
        return hessian

    def build_b_mat(self,xi,xip):
        """
        Given arrays of original points and projected points,
        returns b matrix
        """
        
        xdif = xip[:,0] - xi[:,0]
        x,y = xi[:,0],xi[:,1]
        ydif = xip[:,1] - xi[:,1]

        vsum = np.sum

        b_mat = np.array([
            [vsum(xdif)],
            [vsum(ydif)],
            [vsum(x*xdif)],
            [vsum(y*xdif)],
            [vsum(x*ydif)],
            [vsum(y*ydif)]
        ])
        b_mat = b_mat 
        return b_mat
    
    def get_projection_params(self,xi,xip):
        """
        Given a set of original points and projected points
        returns affine projection parameters as an array
        [tx,ty,a00,a01,a10,a11]

        *a00,a11 still need to be incrementeed by 1.0
        """
        A = self.build_hessian(xi[:,0],xi[:,1])
        self.hessian = A
        b = self.build_b_mat(xi,xip)
        self.b = b

        try:
            p= np.linalg.inv(A)@b
        except:
            p= np.linalg.pinv(A)@b
        p = p[:,0].T
        return p
