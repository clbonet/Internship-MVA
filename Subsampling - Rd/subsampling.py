import numpy as np
import numba

@numba.jit
def filter_func(x,r):
    return x < r


@numba.jit(nopython=True,parallel=True)
def computeConvol(X,gaussVect,f,h,r):

    for i in numba.prange(len(X)):
        x = X[i]
        
        norm = np.sqrt(np.sum((X-x)**2,axis=1))
        convol = 0

        for j in numba.prange(len(norm)):
            if filter_func(norm[j],r):
                convol += gaussVect[j]*h(x-X[j])

        f[i] = convol

    return f


@numba.jit(nopython=True)
def getSample(X,f,b):
    indices = []
    samples = []
    
    for i in numba.prange(len(X)):
        x = X[i]
        
        norm = np.sqrt(np.sum((X-x)**2,axis=1))
        maxi = True
        
        for j in numba.prange(len(norm)):
            if filter_func(norm[j],b) and filter_func(f[i],f[j]):
                maxi = False
                break
        
        if maxi:
            indices.append(i)
            samples.append(x)
            
    return indices,samples


class Subsampler:
    def __init__(self,kernel,b,r):
        self.h = kernel
        self.b = b
        self.r = r
    
    def fit(self,X):
        self.X = X
        self.gaussVect = np.random.randn(len(X))
        self.f = computeConvol(self.X,self.gaussVect,np.zeros(len(X)),self.h,self.r)

    
    def sample(self):                
        """
            Uses numba
        """
        indices,samples = getSample(self.X,self.f,self.b)
        self.indices = np.array(indices)
        self.samples = np.array(samples)
        return self.samples
    
    
    def sample2(self):
        """
            More greedy
            Pick the global max and removes point in the ball of radius b
            Then reiterates without the max
        """
        Y = []
        samples = []
        bools = np.array([True for k in range(len(self.X))])
        f2 = np.copy(self.f)
        
        while(len(self.X[bools])>0):
            argMax = f2.argmax()
            
            Y.append(argMax)
            samples.append(self.X[argMax])
            
            ball = np.linalg.norm(self.X-self.X[argMax],axis=1)<=self.b
            bools[ball] = False
            f2[ball] = -np.inf
        
        self.indices = np.array(Y)
        self.samples = np.array(samples)
        
        return self.samples
    
    
    def sample3(self):
        """
            deletes non max
        """
        Y = []
        samples = []
        bools = np.ones(shape=(self.X.shape[0]))
        
        for ind,x in enumerate(self.X):
            if bools[ind]:
                ball = np.linalg.norm(self.X-x,axis=1)<=self.b
                if not np.any(self.f[ball]>self.f[ind]):
                    Y.append(ind)
                    samples.append(x)
                
                bools[ball][self.f[ball]<self.f[ind]] = 0
                    
        self.indices = np.array(Y)
        self.samples = np.array(samples)
        
        return self.samples
