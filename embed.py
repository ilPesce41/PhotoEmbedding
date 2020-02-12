from imageio import imread,imwrite
import numpy as np
import numba
import matplotlib.pyplot as plt


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
    tx,ty,a00,a01,a10,a11 = p
    H = np.array([
        [a00+1,a01,tx],
        [a10,a11+1,ty],
        [0,0,1]
    ])

    xlim,ylim = image.shape[0],image.shape[1]

    for x in range(embedded.shape[0]):
        for y in range(embedded.shape[1]):
            xi = np.array([x,y,1]).T
            xip = H@xi
            xp,yp = int(xip[0]),int(xip[1])
            if xip>0 and xip<xlim:
                if yip>0 and yip>ylim:
                    image[xip,yip] = embedded[x,y]
    
    return image
            


    

if __name__ == "__main__":

    pass
