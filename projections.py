"""
@author: Cole Hill
University of South Florida
Department of Computer Science and Engineering
Computer Vision
Spring 2020
"""
import numpy as np
class Affine:
    """
    Class for estimating the affine projection
    relating an array of points `points` to an
    array of points `projected`
    """
    def __init__(self,points,projected):

        self.points = points
        self.projected = projected
        #Estimate projection parameters
        self.p = self.get_projection_params(points,projected)

    def build_hessian(self,x,y):
        """
        Takes array of x and y points in original image,
        returns hessian for affine projection
        """
        #Precalculate entries of hessian
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
        return hessian

    def build_b_mat(self,xi,xip):
        """
        Given arrays of original points and projected points,
        returns b matrix
        """

        #Precalculate entries
        xdif = xip[:,0] - xi[:,0]
        x,y = xi[:,0],xi[:,1]
        ydif = xip[:,1] - xi[:,1]
        vsum = np.sum
        #Define b matrix
        b_mat = np.array([
            [vsum(xdif)],
            [vsum(ydif)],
            [vsum(x*xdif)],
            [vsum(y*xdif)],
            [vsum(x*ydif)],
            [vsum(y*ydif)]
        ])
        return b_mat
    
    def get_projection_params(self,xi,xip):
        """
        Given a set of original points and projected points
        returns affine projection parameters as an array
        [tx,ty,a00,a01,a10,a11]

        *a00,a11 still need to be incrementeed by 1.0
        """
        #Calculate Hessian and b vector
        A = self.build_hessian(xi[:,0],xi[:,1])
        self.hessian = A
        b = self.build_b_mat(xi,xip)
        self.b = b

        #Calculate parameters
        try:
            p= np.linalg.inv(A)@b
        except:
            p= np.linalg.pinv(A)@b
        p = p[:,0].T
        tx,ty,a00,a01,a10,a11 = p
        #Save projection matrix
        self.H = np.array([
            [1+a00,a01,tx],
            [a10,1+a11,ty],
            [0,0,1]
        ])
        return p


class Homogeneous:
    """
    Class for estimating the homogeneous projection
    relating an array of points `points` to an
    array of points `projected`
    """
    def __init__(self,points,projected):

        self.points = points
        self.projected = projected

        #Create Affine projection as initial estimate
        self.affine = Affine(points,projected)
        #Estimate Homogeneous parameters
        self.get_projection_params(points,projected)
    
    def build_hessian(self,x,y,x_est,y_est,H_est):
        """Calculate Hessian"""
        
        #Precalculate some entries
        h20 = H_est[2,0]
        h21 = H_est[2,1]
        D = h20*x + h21*y + 1
        xp = x_est
        yp = y_est
        
        hessian = np.array([
        [       x**2/(D)**2,     (x*y)/(D)**2,     (1*x)/(D)**2,                               0,                               0,                               0,           -(x**2*xp)/(D)**2,           -(x*xp*y)/(D)**2],
        [     (x*y)/(D)**2,       y**2/(D)**2,     (1*y)/(D)**2,                               0,                               0,                               0,           -(x*xp*y)/(D)**2,           -(xp*y**2)/(D)**2],
        [     (1*x)/(D)**2,     (1*y)/(D)**2,       1**2/(D)**2,                               0,                               0,                               0,           -(1*x*xp)/(D)**2,           -(1*xp*y)/(D)**2],
        [                               0,                               0,                               0,       x**2/(D)**2,     (x*y)/(D)**2,     (1*x)/(D)**2,           -(x**2*yp)/(D)**2,           -(x*y*yp)/(D)**2],
        [                               0,                               0,                               0,     (x*y)/(D)**2,       y**2/(D)**2,     (1*y)/(D)**2,           -(x*y*yp)/(D)**2,           -(y**2*yp)/(D)**2],
        [                               0,                               0,                               0,     (1*x)/(D)**2,     (1*y)/(D)**2,       1**2/(D)**2,           -(1*x*yp)/(D)**2,           -(1*y*yp)/(D)**2],
        [ -(x**2*xp)/(D)**2, -(x*xp*y)/(D)**2, -(1*x*xp)/(D)**2, -(x**2*yp)/(D)**2, -(x*y*yp)/(D)**2, -(1*x*yp)/(D)**2, (x**2*(xp**2 + yp**2))/(D)**2, (x*y*(xp**2 + yp**2))/(D)**2],
        [ -(x*xp*y)/(D)**2, -(xp*y**2)/(D)**2, -(1*xp*y)/(D)**2, -(x*y*yp)/(D)**2, -(y**2*yp)/(D)**2, -(1*y*yp)/(D)**2, (x*y*(xp**2 + yp**2))/(D)**2, (y**2*(xp**2 + yp**2))/(D)**2]
         ])
        #Sum up elements of hessian matrix
        for i in range(hessian.shape[0]):
            for j in range(hessian.shape[1]):
                hessian[i,j] = np.sum(hessian[i,j])
        hessian = hessian.astype(float)
        return hessian
        
    
    def build_b_mat(self,x,y,x_,y_,xp,yp,H_est): 
        """
        Calculates b matrix
        """
        #Precalculate denominator
        h20 = H_est[2,0]
        h21 = H_est[2,1]
        D = h20*x + h21*y + 1
        
        self.b = np.array([
            (x*(x_ - xp))/(D),
            (y*(x_ - xp))/(D),
            (1*(x_ - xp))/(D),
            (x*(y_ - yp))/(D),
            (y*(y_ - yp))/(D),
            (1*(y_ - yp))/(D),
            - (x*xp*(x_ - xp))/(D) - (x*yp*(y_ - yp))/(D),
            - (xp*y*(x_ - xp))/(D) - (y*yp*(y_ - yp))/(D)
        ])
        #Sum up entries
        self.b = np.sum(self.b,axis=1)
        return self.b
        

    def get_projection_params(self,xi,xip):
        """
        Gets projection matrix for two points
        """
        #Use affine projection as estimate
        H_est = self.affine.H
        self.H = H_est
        #Convert to Homogeneous coords
        xip = np.vstack([xip.T,np.ones(xip.shape[0])]).T
        
        #Get projection estimate
        xp_est = []
        xi = np.vstack([xi.T,np.ones(xi.shape[0])]).T
        for i in range(xi.shape[0]):
            xp_est.append(H_est@xi[i])
        
        xp_est = np.array(xp_est)
        xp_est = (xp_est.T/xp_est[:,-1]).T

        #Establish error
        eps = np.sum(np.abs(xp_est-xip))
        k=0
        #Use Gauss-Newton non-linear least squares optimization
        while eps>.01 and k<1000:
            k += 1

            #Calculate Hessian an b vector
            self.build_b_mat(xi[:,0],xi[:,1],xip[:,0],xip[:,1],xp_est[:,0],xp_est[:,1],H_est)
            A = self.build_hessian(xi[:,0],xi[:,1],xp_est[:,0],xp_est[:,1],H_est)
            #Calculate parameter delta
            p_est = np.linalg.pinv(A.astype(float))@self.b
            H_est = np.hstack([p_est,[0]]).reshape(3,3)
            
            #Add parameter delta to current paramters
            self.H = self.H + H_est
            H_est = self.H

            #Calculate error
            xp_est = []
            for i in range(xi.shape[0]):
                xp_est.append(H_est@xi[i])
            xp_est = np.array(xp_est)
            xp_est = (xp_est.T/xp_est[:,-1]).T
            eps = np.sum(np.abs(xp_est-xip))

        
if __name__ == "__main__":

    pass