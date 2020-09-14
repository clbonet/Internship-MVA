import numba
import numpy as np

@numba.jit
def filter_func(x,r):
    return x < r


@numba.jit(nopython=True,parallel=True)
def computeConvol(A,gaussVect,f,h):
    """
        Convolution on neighbours
    """
    for i in numba.prange(A.shape[0]):
        convol = 0
        for j in numba.prange(A.shape[0]):
            if A[i,j]: ## neighbours
                convol += gaussVect[j]*h[i,j] ##
        f[i] = convol
    return f


@numba.jit(nopython=True,parallel=True)
def computeConvol2(gaussVect,f,h):
    """
        Convolution on all nodes
    """
    for i in numba.prange(h.shape[0]):
        convol = 0
        for j in numba.prange(h.shape[0]):
            convol += gaussVect[j]*h[i,j] ##
        f[i] = convol
    return f


@numba.jit(nopython=True)
def bubbleSort(f):
    for i in range(len(f)):
        for j in range(0, len(f)-i-1):
            if f[j] < f[j+1] :
                f[j], f[j+1] = f[j+1], f[j]
                
    return f


@numba.njit
def getSample(A,f,k):
    indices = []
    for i in numba.prange(A.shape[0]):
        neighbours = [j for j in range(A.shape[0]) if A[i,j]]
        val = f[i]
        f2 = [f[j] for j in neighbours]
        if len(f2)>0:
            val_neighbours = bubbleSort(f2) # np.sort(f[neighbours])[::-1]
            if filter_func(val_neighbours[int(len(val_neighbours)*k)],val):
                indices.append(i)
        else:
            indices.append(i)

    return indices


@numba.jit
def filter_func(x,r):
    return x < r


@numba.jit(nopython=True)
def getSample2(f,b,K,p=0.1):
    """
        keep points if belong to p*100% better points in its ball of radius b (with norm K)
    """
    indices = []
    
    for i in numba.prange(len(f)):        
        norm = K[i,:]
        
        f2 = [f[j] for j in range(len(f)) if norm[j]<=b]
        vals = bubbleSort(np.array(f2))
        
        if vals[int(len(vals)*p)]<=f[i]:
            indices.append(i)
            
    return indices


class Subsampler():
    def __init__(self,kernel,dist=None):
        """
            kernel : function which takes an adjacency matrix as input
            dist: dist function, eg euclidean_distances
        """
        self.kernel = kernel
        self.dist = dist
    
    def fit(self,adjacency,expectation=0,variance=1):
        self.h = self.kernel(adjacency)
        if self.dist is not None:
            self.d = self.dist(adjacency,adjacency)
        self.A = adjacency
        self.gaussVect = np.sqrt(variance)*np.random.randn(self.A.shape[0])+expectation*np.ones(self.A.shape[0])
        self.f = computeConvol(self.A@np.eye(self.A.shape[1]),self.gaussVect,np.zeros(self.A.shape[0]),self.h)
        
    def fit2(self,adjacency,expectation=0,variance=1):
        self.h = self.kernel(adjacency)  
        self.A = adjacency
        self.gaussVect = np.sqrt(variance)*np.random.randn(self.A.shape[0])+expectation
        self.f = computeConvol2(self.gaussVect,np.zeros(self.A.shape[0]),self.h)    
        
        
    def sample(self,k=0.1):
        """
            keep k*100% best points in neighbourhood (using numba)
        """
        n = self.A.shape[1]
        self.samples = np.array(getSample(self.A@np.eye(n),self.f,k))
        return self.samples
    
    def sample2(self,k=0.1):
        """
            keep k*100% best points in neighbourhood using ball of the kernel (using numba)
        """
        n = self.A.shape[1]
        b = np.percentile(self.d,k*100)
        self.samples = np.array(getSample2(self.f,b,self.d,k))
        return self.samples            
        
    def sample3(self):
        """
            Neighborhood Sampling
        """
        indices = getSample(self.A@np.eye(self.A.shape[1]),self.f)
        
        indices2 = indices[:]
        for i,ind in enumerate(indices):
            for j in range(self.A.shape[0]):
                if self.A[ind,j] and j not in indices2:
                    indices2.append(j)
                    
        self.samples = np.array(indices2)
        return self.samples
        
