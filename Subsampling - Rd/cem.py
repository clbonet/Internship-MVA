import numba
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import k_means
from copy import deepcopy

from subsampling import *
from utils_EM import *


@numba.jit(nopython=True,fastmath=True)
def logNormal(x,mu,sigma):
    d = len(x)
    diff = (x-mu).reshape(-1,1)
    inv_sig = np.linalg.inv(sigma)
    det_sig = np.linalg.det(sigma)
    return (-(1/2)*diff.T@inv_sig@diff - (d/2)*np.log(2*np.pi) - (1/2)*np.log(det_sig))[0][0]


@numba.jit(nopython=True,parallel=True)
def C2(X,K,alphas,mus,sigmas,classes):
    n,d = X.shape
    cpt = 0
    for k in numba.prange(K):
        for i in numba.prange(n):
            if classes[i] == k:
                cpt += np.log(alphas[k]) + logNormal(X[i],mus[k],sigmas[k])
    return cpt


class CEM():
    """
        A classification EM algorithm for clustering and two stochastic versions, Celeux et Govaert, 2006
        https://hal.inria.fr/inria-00075196/document
    """
    def __init__(self,n_components=1,tol=1e-6,reg_covar=1e-6,n_iter=100):
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        
        self.reg_covar = reg_covar
        
    def C2(self):
        return C2(self.X,self.n_components,np.array(self.alphas),np.array(self.mus), \
                  np.array(self.sigmas),np.array(self.classes))
    
    def E_Step(self):
        n,d = self.X.shape
        m = self.n_components
        for k in range(m):
            for i in range(n):
                self.P[i,k] = self.alphas[k] * np.exp(logNormal(self.X[i],self.mus[k],self.sigmas[k]))
        for i in range(n):
            self.P[i,:] /= np.sum(self.P[i,:])
    
    def C_Step(self):
        n,d = self.X.shape
        self.classes = []
        for i in range(n):
            self.classes.append(self.P[i,:].argmax())
    
    def M_step(self):
        n,d = self.X.shape
        m = self.n_components

        for k in range(m):
            
            pointsInK = [i for i in range(n) if self.classes[i]==k]
            
            ## avoid empty cluster
            if len(pointsInK)==0:
                pointsInK.append(np.random.randint(n))
                self.classes[pointsInK[0]] = k
                print("Empty Cluster")
            
            self.alphas[k] = len(pointsInK)/n
            self.mus[k] = np.sum(self.X[pointsInK],axis=0)/len(pointsInK)
            
            cpt = np.zeros(self.sigmas[k].shape)
            for i in pointsInK:
                diff = (self.X[i]-self.mus[k]).reshape(-1,1)
                cpt += diff@diff.T
            
            self.sigmas[k] = cpt/len(pointsInK) + 1e-6*np.eye(self.sigmas[k].shape[0]) # avoid singular matrix
    
    def fit(self,X,alphas0,mus0,sigmas0,plot=False,stop=True):
        self.X = X
        
        assert len(alphas0) == self.n_components, "Wrong size"
        assert len(mus0) == self.n_components, "Wrong size"
        assert len(sigmas0) == self.n_components, "Wrong size"
        
        self.alphas = alphas0
        self.mus = mus0
        self.sigmas = sigmas0
        self.P = np.ones((len(X),self.n_components))
        
        L = []
        
        for i in range(self.n_iter):
            self.E_Step()
            self.C_Step()
            self.M_step()
            
            L.append(self.C2())
            
            if plot:
                print("C2",L[-1])
        
            if(stop and len(L)>1 and L[-1]-L[-2]<self.tol):
                break
    
        if plot:
            plt.plot(range(len(L)),L,'ro')
            plt.title("$C_2$ over iterations")
            plt.xlabel("Iterations")
            plt.ylabel("$C_2$")
            plt.show()
            
            
def getClasses(X,inds,cem,classes):
    """
        return classes of all samples (subsample used to init the CEM and other sample used sequentially
        for the online CEM)
        
        Input:
        - X: full data
        - inds: indices of points used in the initial CEM
        - cem: CEM object used in the online CEM
        - classes: classes returned in the online setting of online CEM
        
        Output:
        - numpy array containing all the classes in the order of appearance in X
    """
    classes2 = []

    k_X0 = 0
    k_XX = 0

    for i in range(len(X)):
        if i in inds:
            classes2.append(cem.classes[k_X0])
            k_X0 += 1
        else:
            classes2.append(classes[k_XX])
            k_XX += 1
            
    return np.array(classes2)


def plotOnlineCEM(X,K,alphas,mus,sigmas,inds,classes,cem,printClasses=False,printARI=False,labels=None):
    """
        plot the results of the initial cem and of the online cem
        
        Input:
        - alphas: weights
        - mus: list of centers
        - sigmas: list of covariances
        - inds: indices of points used in the initial CEM
        - cem: CEM object used for the online CEM
        - printClasses: boolean (default False) : if True, print the the number of sample in each cluster
        - printARI: boolean (default False) : if True, print the ARI index, requires to fill labels
        (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)
        - labels: list of true labels (default None), required to print the ARI
    """
    classes2 = getClasses(X,inds,cem,classes)

    if printARI and labels is not None:
        print("ARI",adjusted_rand_score(labels,classes2))

    if printClasses:
        for k in range(K):
            print(k,len([i for i in range(len(X)) if classes2[i]==k]))

    fig,ax = plt.subplots(1,2,figsize=(15,7))

    plot_ellipses(0,1,ax[0],K,cem.mus,cem.sigmas,cem.alphas,dico=cem.classes,X=cem.X)
    plot_ellipses(0,1,ax[1],K,mus,sigmas,alphas,dico=classes2,X=X)
    ax[1].scatter(cem.X[:,0],cem.X[:,1],c='purple')
    plt.show()

    
def onlineCEM(X,X0,alphas0,mus0,sigmas0,K,plot=False,stop=True,n_iter=100):
    """
        An on-line Classification EM algorithm based on mixture model, Samé et al, 2007 : 
        https://www.researchgate.net/publication/220286457_An_online_classification_EM_algorithm_based_on_the_mixture_model
        
        Inputs:
        - X: data points for the online part of the algorithm
        - X0: data points for the CEM part of the algorithm
        - alphas0: Initial weights (for example (1/K,...,1/K))
        - mus0: Initial centers
        - sigmas0: Initial covariances
        - plot: boolean (default False), if True then plot the curve C2 of the CEM
        - stop: boolean (default True), if True then the first CEM stops whenever C2 does not increase anymore
        - n_iter: number of iterations for the CEM
        
        Outputs:
        - alphas: weights returned by online CEM
        - mus: centers returned by online CEM
        - sigmas: covariances returned by online CEM
        - classes: classes returned by online CEM
        - cem: CEM object after the initial CEM
    """
    ## Initial CEM
    cem = CEM(K,n_iter=n_iter)
    cem.fit(X0,alphas0,mus0,sigmas0,plot=plot,stop=stop)
    
    n0 = len(X0)
    K = len(alphas0)
    
    P = np.ones((len(X),K))
    classes = []
    
    alphas = cem.alphas.copy()
    mus = cem.mus.copy()
    sigmas = deepcopy(cem.sigmas)
    
    n_clusters = []
    for k in range(K):
        n_clusters.append(len([i for i in range(len(X0)) if cem.classes[i]==k]))
        
    ## Online CEM 
    for n in range(len(X)):
        ## if n not in X0: # ?
        for k in range(K):
            P[n,k] = alphas[k]*np.exp(logNormal(X[n],mus[k],sigmas[k]))
        P[n,:] /= np.sum(P[n,:])
        
        ind = np.argmax(P[n,:])
        classes.append(ind)
        n_clusters[ind] += 1
        
        for k in range(K):
            diff = (X[n]-mus[k]).reshape(-1,1)
            if ind==k:
                alphas[k] = (n+n0)/(n+n0+1) * alphas[k] + 1/(n+n0+1)
                
                mus[k] = mus[k] + (X[n]-mus[k])/n_clusters[ind]
                sigmas[k] += ((1-1/n_clusters[ind])*diff@diff.T - sigmas[k])/n_clusters[ind]
            else:
                alphas[k] = (n+n0)/(n+n0+1) * alphas[k]
                
    return alphas,mus,sigmas,classes,cem


def CEM_KMeans(X,K,plot=False,stop=True,init='k-means++'):
    """
        CEM with K-means initialization
        
        Inputs:
        - X: Full data
        - K: number of clusters
        - plot: Boolean (by default False) : if True then plot the C2 curve of the CEM
        - stop: Boolean (By default True) : if True then stop the CEM whenever C2 does not increase
        - init: {‘k-means++’, ‘random’, ndarray}, default=’k-means++’ 
        (see sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
        
        Output:
        - CEM object
    """
    alphas0 = np.ones(K)/K
    sigmas0 = [np.eye(X.shape[1])/K for l in range(K)]
    mus0,_,_ = k_means(X,K,max_iter=10,n_init=10,init=init) 
    
    cem = CEM(K,n_iter=50)
    cem.fit(X,alphas0,mus0,sigmas0,plot=plot,stop=stop)
    
    return cem


def uniformOnlineCEM_KMeans(X,n0,K,plot=False,stop=True,init='k-means++'):
    """
        Online CEM with K-means initialization and uniform subsampling
        
        Inputs:
        - X: Full data
        - n0: number of subsample for the initial CEM
        - K: number of clusters
        - plot: Boolean (by default False) : if True then plot the C2 curve of the CEM
        - stop: Boolean (By default True) : if True then stop the CEM whenever C2 does not increase
        - init: {‘k-means++’, ‘random’, ndarray}, default=’k-means++’ 
        (see sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
        
        Output:
        - alphas : weights returned by online CEM
        - mus: centers returned by online CEM
        - sigmas: covariances returned by online CEM
        - inds: indices of the subsample
        - classes: classes returned by online CEM
        - cem: CEM object from the CEM part
    """
    inds = list(set(np.random.randint(len(X),size=n0))) ## avoid duplicates???
    X0 = X[inds]
    XX = X[[i for i in range(len(X)) if i not in inds]]
    
    alphas0 = np.ones(K)/K
    sigmas0 = [np.eye(X.shape[1])/K for l in range(K)]
    mus0,_,_ = k_means(X0,K,max_iter=10,n_init=10,init=init) 
    
    alphas,mus,sigmas,classes,cem = onlineCEM(XX,X0,alphas0,mus0,sigmas0,K,plot,stop=stop,n_iter=50)
    return alphas,mus,sigmas,inds,classes,cem


def subsamplingOnlineCEM_KMeans(X,h,b,r,K,plot=False,stop=True,init='k-means++'):
    """
        Online CEM with K-means initialization and subsampling via our algorithm
        
        Inputs:
        - X: Full data
        - h: kernel for the algorithm of subsampling
        - b: radius for the local maxima
        - r: radius for the convolutions
        - K: number of clusters
        - plot: Boolean (by default False) : if True then plot the C2 curve of the CEM
        - stop: Boolean (By default True) : if True then stop the CEM whenever C2 does not increase
        - init: {‘k-means++’, ‘random’, ndarray}, default=’k-means++’ 
        (see sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
        
        Output:
        - alphas : weights returned by online CEM
        - mus: centers returned by online CEM
        - sigmas: covariances returned by online CEM
        - inds: indices of the subsample
        - classes: classes returned by online CEM
        - cem: CEM object from the CEM part
        - subsampler: Subsampler object used to subsample
    """
    
    subsampler = Subsampler(h,b,r)
    subsampler.fit(X)
    X0 = subsampler.sample()
    inds = subsampler.indices
    
    XX = X[[i for i in range(len(X)) if i not in inds]]
    
    alphas0 = np.ones(K)/K
    sigmas0 = [np.eye(X.shape[1])/K for l in range(K)]
    mus0,_,_ = k_means(X0,K,max_iter=10,n_init=10,init=init) 
    
    alphas,mus,sigmas,classes,cem = onlineCEM(XX,X0,alphas0,mus0,sigmas0,K,plot,stop=stop,n_iter=50)
    return alphas,mus,sigmas,inds,classes,cem,subsampler