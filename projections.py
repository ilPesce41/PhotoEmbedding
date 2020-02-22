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
        tx,ty,a00,a01,a10,a11 = p
        self.H = np.array([
            [1+a00,a01,tx],
            [a10,1+a11,ty],
            [0,0,1]
        ])
        return p


class Homogeneous:

    def __init__(self,points,projected):

        self.points = points
        self.projected = projected
        self.affine = Affine(points,projected)
        self.get_projection_params(points,projected)
    
    def build_hessian(self,x,y,x_est,y_est,H_est):
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
        for i in range(hessian.shape[0]):
            for j in range(hessian.shape[1]):
                hessian[i,j] = np.sum(hessian[i,j])
        hessian = hessian.astype(float)
        return hessian
        
    
    def build_jacobian(self,x,y,x_est,y_est,H_est):

        vsum = np.sum
        D = x*H_est[2,0] + y*H_est[2,1] + 1
        self.J = np.array([
            [x/D, y/D, len(x)/D, 0,0,0, (-x*x_est)/D, (-y*x_est)/D],
            [0,0,0,x/D, y/D, len(x)/D, (-x*y_est)/D, (-y*y_est)/D]
        ])
        for i in range(self.J.shape[0]):
            for j in range(self.J.shape[1]):
                self.J[i,j] = vsum(self.J[i,j])
        return self.J.astype(float)

    def build_b_mat(self,x,y,x_,y_,xp,yp,H_est): 
        h20 = H_est[2,0]
        h21 = H_est[2,1]
        D = h20*x + h21*y + 1
        # 1 = len(x)
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
        self.b = np.sum(self.b,axis=1)
        #print("DIFFX",xp-x_est)
        #print(self.b)
        return self.b
        

    def get_projection_params(self,xi,xip):
        
        H_est = self.affine.H
        self.H = H_est
        xip = np.vstack([xip.T,np.ones(xip.shape[0])]).T
        
        xp_est = []
        xi = np.vstack([xi.T,np.ones(xi.shape[0])]).T
        for i in range(xi.shape[0]):
            xp_est.append(H_est@xi[i])
        
        xp_est = np.array(xp_est)
        xp_est = (xp_est.T/xp_est[:,-1]).T

        eps = np.sum(np.abs(xp_est-xip))
        print(eps)

        k=0
        while eps>.01 and k<1000:
            k += 1
            self.build_b_mat(xi[:,0],xi[:,1],xip[:,0],xip[:,1],xp_est[:,0],xp_est[:,1],H_est)
            print(self.b)
            A = self.build_hessian(xi[:,0],xi[:,1],xp_est[:,0],xp_est[:,1],H_est)
            # #print(A)
            p_est = np.linalg.pinv(A.astype(float))@self.b
            # print("deltap",p_est)
            print(p_est)
            H_est = np.hstack([p_est,[0]]).reshape(3,3)
            print(H_est)
            # print()
            self.H = self.H + H_est
            H_est = self.H
            # print(self.H)
            # raise Exception
            xp_est = []
            for i in range(xi.shape[0]):
                xp_est.append(H_est@xi[i])
            xp_est = np.array(xp_est)
            xp_est = (xp_est.T/xp_est[:,-1]).T
            eps = np.sum(np.abs(xp_est-xip))
            print(eps)
            # if k%5==0:
            #     # print(eps)
            #     raise Exception
            #print(eps)
            #print(xip)
            #print(xp_est)
            # if k ==2:
            #     raise Exception

            




if __name__ == "__main__":

    imm1 = np.array([
        [0, 153], 
        [0, 153], 
        [153, 0], 
        [153, 153]
    ])

    imm2 = np.array([
        [409, 247], 
        [425, 410], 
        [665, 237], 
        [673, 422]
    ])

    proj = Homogeneous(imm1,imm2)
    proj.H
    
    for i in range(len(imm1)):
        arr = np.hstack([imm1[i],[1]]).T
        #print(proj.H@arr)