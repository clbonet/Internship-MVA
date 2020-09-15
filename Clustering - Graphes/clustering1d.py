import numba
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import k_means
from sknetwork.clustering import modularity


@numba.njit
def np_apply_along_axis(func1d, axis, arr):
    """
        https://github.com/numba/numba/issues/1269
    """
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result

@numba.njit
def np_mean(array, axis):
    return np_apply_along_axis(np.mean, axis, array)

@numba.njit
def np_std(array, axis):
    return np_apply_along_axis(np.std, axis, array)

@numba.jit(nopython=True, parallel=True)
def iterate(f_k,adjacency_indptr,adjacency_indices,d):
    n = adjacency_indptr.shape[0]-1
    
    f = np.zeros((n,d))
    
    for i in numba.prange(n):
        row_start = adjacency_indptr[i]
        row_end = adjacency_indptr[i+1]
        neighbours = adjacency_indices[row_start:row_end]
        f[i,:] = np_mean(f_k[neighbours],axis=0) # np.mean(f_k[neighbours],axis=0)
    
    return f

def getIterateAverages(adjacency,n=1000,mu=1,sigma=1/2,stop=True,plot=False,epsilon=1e-5,d=1,K=2):
    """
        Inputs:
        - adjacency: adjaceny matrix
        - n: number of iterations (default 1000)
        - mu: mean of initial gaussian variable (default 1)
        - sigma: variance of initial gaussian variable (default 1/2)
        - stop: boolean (default True): stop the iterations using variance criterion
        - plot: boolean (default False): plot of variance and modularity obtained by kmeans
        - epsilon: stopping criterion for variance
        - d: dimension f0
        - K: number of cluster k-means
    """
    f0 = np.sqrt(sigma)*np.random.randn(adjacency.shape[0],d)+mu

    fs = [f0]
    var = []
    
    if plot:
        mod = []

    for k in range(n):
        
        fs.append(iterate(fs[-1],adjacency.indptr,adjacency.indices,d))
        
        var.append(np.sum(np.std(fs[-1],axis=0)**2))
        
        if plot:
            c,labels_k,inertia = k_means(fs[-1],K)
            mod.append(modularity(adjacency,labels_k))

        if(stop and k>1 and np.abs(var[-2]-var[-1])<epsilon):
            break
            
    if plot:
        plt.plot(var)
        plt.xlabel("iterations")
        plt.ylabel(r"$\sigma^2$")
        plt.title("Variance")
        plt.show()
        
        plt.plot(mod)
        plt.xlabel("iterations")
        plt.ylabel("Q")
        plt.title("Modularity")
        plt.show()
            
    return fs
