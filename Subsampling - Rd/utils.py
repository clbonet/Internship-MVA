import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def generateMixture(n,mus,sigmas,ps):
    """
        Generates points of a GMM with centers mus, covariances sigmas and weights ps
        
        Input:
        - n: number of points
        - mus: list/array of centers
        - sigmas: list of covariances matrices
        - ps: list of weights
        
        Output:
        - X: numpy array of n points
        - y: numpy array of associated labels
    """
    Z = np.random.multinomial(n,ps)
    X = []
    y = []
    for k in range(len(Z)):
        normals = np.random.multivariate_normal(mus[k],sigmas[k],size=Z[k])
        X += list(normals)
        y += [k for i in range(len(normals))]
    return np.array(X),np.array(y)


class GaussianMixture():
    def __init__(self, p, mus, sigmas):
        self.means = mus
        self.covariances = sigmas
        self.weights = p
        self.rvs = [multivariate_normal(mus[k], sigmas[k]) for k in range(len(self.weights))]

    def pdf(self, X):
        return np.sum([self.weights[i]*self.rvs[i].pdf(X) for i in range(len(self.weights))], axis=0)

    
def plotContour(gm):
    """
        Plot contour of a GaussianMixture object
        
        Input:
        - gm : Gaussian Mixture Object
    """
    stdsig = 2
    xx, yy = np.mgrid[np.min(
        gm.means[:, 0])-stdsig:np.max(gm.means[:, 0])+stdsig:.1, np.min(gm.means[:, 1])-stdsig:np.max(gm.means[:, 1])+stdsig:.1]
    pos = np.empty(xx.shape + (2,))
    pos[:,:,0] = xx; pos[:,:,1] = yy

    f = gm.pdf(pos)
    plt.figure(figsize=(10, 7))
    plt.contour(xx, yy, f, 20)
    # plt.scatter(*mus.T, facecolors='none', edgecolors='r')

    plt.show()
    
    
def plotDensity(density):
    """
        plot the density
        
        Input:
        - density: function returning the density
    """
    N = 500
    z = np.zeros([N,N])
    x = np.zeros([N,N])
    y = np.zeros([N,N])
    xmin = -5
    xmax = 5
    ymin = -5
    ymax = 5

    x1 = np.linspace(xmin,xmax,N)
    y1 = np.linspace(ymin,ymax,N)
    for i in range(N):
        for j in range(N):
            x[i,j] = x1[i]
            y[i,j] = y1[j]
            z[i,j] = density([x1[i],y1[j]])

    return x,y,z
    
    
def plotSubSample(subsampler):
    """
        scatter the initial sample and the subsample
        
        Input:
        - subsampler : Object Subsampler
    """
    fig,ax = plt.subplots(1,2,figsize=(15,7))

    f = np.array(subsampler.f)
    
    cb0 = ax[0].scatter(subsampler.X[:,0],subsampler.X[:,1],c=f)
    ax[0].set_title("f on Initial Sample")
    cb = ax[1].scatter(subsampler.samples[:,0],subsampler.samples[:,1],c=f[subsampler.indices])
    ax[1].set_title("f on sample")
    fig.colorbar(cb,ax=ax,location="right")
    # fig.colorbar(cb0,ax=ax,location="left")
    plt.show()