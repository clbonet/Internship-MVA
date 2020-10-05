import numpy as np
import matplotlib.pyplot as plt

import numba

from sklearn.cluster import k_means
from sknetwork.clustering import KMeans
from sknetwork.embedding import GSVD
from copy import deepcopy


eps = np.finfo(float).eps
xmin = np.finfo(np.float).min
xmax = np.finfo(np.float).max


@numba.jit(nopython=True,parallel=True)
def J(X,taus,alphas,pis,Q):
    n = X.shape[0]

    output = np.sum(taus@np.log(alphas))

    cpt = 0
    for i in numba.prange(n):
        for j in numba.prange(n):
            if i!=j:
                for q in numba.prange(Q):
                    for l in numba.prange(Q):
                        logb = (X[i,j]*np.log(pis[q,l])+ \
                               (1-X[i,j])*np.log(1-pis[q,l]))
                        cpt += taus[i,q]*taus[j,l]*logb

    return output+cpt/2-np.sum(taus*np.log(taus))


@numba.jit(nopython=True,parallel=True)
def E_step_VEM(X,taus,alphas,pis,Q):
    n = X.shape[0]
    logTau = np.log(np.maximum(taus,eps))
    for i in numba.prange(n):
        logTau[i,:] = np.log(alphas)
        for q in numba.prange(Q):
            for j in numba.prange(n):
                if j!=i:
                    for l in numba.prange(Q):
                        logTau[i,q] += taus[j,l]*(X[i,j]*np.log(pis[q,l])+\
                                                 (1-X[i,j])*np.log(1-pis[q,l]))

    logTau = np.maximum(np.minimum(logTau,xmax),xmin)
    tau = np.exp(logTau)

    for i in numba.prange(n):
        tau[i,:] /= np.sum(tau[i,:])

    return np.maximum(tau,eps)


@numba.jit(nopython=True,parallel=True)
def M_step_VEM(X,taus,alphas,pis,Q):
    n = X.shape[0]
    alphas = np.maximum(np.sum(taus,axis=0)/n,eps)

    for q in numba.prange(Q):
        for l in range(Q):
            num = 0
            denom = 0
            for i in numba.prange(n):
                for j in numba.prange(n):
                    if i!=j:
                        num += taus[i,q]*taus[j,l]*X[i,j]
                        denom += taus[i,q]*taus[j,l]

            if denom>eps:
                pi = num/denom
            else:
                ## class with a single vertex
                pi = 0.5
                
            pis[q,l] = np.minimum(np.maximum(pi,eps),1-eps)
    
    return alphas


class VEM():
    def __init__(self,Q):
        self.Q = Q
        
    def init_vem(self,init):
        n = self.X.shape[0]
        
        if init=="k-means":
            # kmeans = KMeans(n_clusters = self.Q, embedding_method=GSVD(self.Q))
            # labels_k = kmeans.fit_transform(self.X)
            
            embedding = GSVD(self.Q).fit_transform(self.X)
            centroid,labels_k,inertia = k_means(embedding,self.Q,n_init=5)

            self.taus = np.zeros(shape=(n,self.Q))
            self.taus[:] = np.eye(self.Q)[labels_k]
        else:
            self.taus = np.random.rand(n,self.Q)
            for i in range(n):
                self.taus[i,:] /= np.sum(self.taus[i,:])
            
        self.alphas = np.ones(shape=(self.Q,))/self.Q
        self.pis = np.zeros(shape=(self.Q,self.Q))
        
        
    def J(self):
        n = self.X.shape[0]
        return J(self.X@np.eye(n),self.taus,self.alphas,self.pis,self.Q)
                            
    
    def E_step(self):
        n = self.X.shape[0]
        self.taus = E_step_VEM(self.X@np.eye(n),self.taus,self.alphas,self.pis,self.Q)
    
    
    def M_step(self):
        n = self.X.shape[0]
        self.alphas = M_step_VEM(self.X@np.eye(n),self.taus,self.alphas,self.pis,self.Q)           
                
    
    def fit(self,adjacency,init="k-means",n_iter=100,plot=False):
        self.X = deepcopy(adjacency)        
        n = self.X.shape[0]
        
        self.init_vem(init)
        
        Js = []
        
        for k in range(n_iter):
            self.M_step()
            
            if plot:
                J1 = self.J()
                if len(Js)>1 and J1<Js[-1]:
                    print("Error M-step", J1-Js[-1])
                    
            self.E_step()
            
            Js.append(self.J())
            
            if plot:
                print(k,Js[-1])
                if Js[-1]<J1:
                    print("Error E-step",Js[-1]-J1)
                
            if len(Js)>1 and Js[-1]<Js[-2]:
                break
                print("Error",Js[-2]-Js[-1])
            elif len(Js)>1 and Js[-1]-Js[-2]<1e-6:
                break
                    
        if plot:
            plt.plot(range(len(Js)),Js,'ro')
            plt.title("$\mathcal{J}(R_{\mathcal{X}})$ over iterations")
            plt.xlabel("Iterations")
            plt.ylabel("$\mathcal{J}(R_{\mathcal{X}})$")
            plt.show()
                    
        self.labels = np.argmax(self.taus,axis=1)
        return self.labels
