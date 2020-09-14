import numba

import numpy as np

from copy import deepcopy

from sknetwork.clustering import KMeans
from sknetwork.embedding import GSVD

eps = np.finfo(float).eps
xmin = np.finfo(np.float).min
xmax = np.finfo(np.float).max


@numba.jit(nopython=True)
def complete_log_likelihood(inds,X,z,alphas,pis):
    output = np.sum(z@np.log(alphas))
    
    for cpti,i in enumerate(inds):
        q = np.argmax(z[i,:])
        for cptj in range(cpti):
            j = inds[cptj]
            l = np.argmax(z[j,:])            
            output += (X[i,j]*np.log(pis[q,l])+\
                      (1-X[i,j])*(np.log(1-pis[q,l])))
            
    return output
    

@numba.jit(nopython=True)
def partial_log_likelihood(m,X,z,alphas,pis,inds):
    n = X.shape[0]
    
    q = np.argmax(z[m,:])
    z[m,:] = 0
    
    output = complete_log_likelihood(inds,X,z,alphas,pis)
            
    output += np.log(alphas[q])
    for j in inds:
        l = np.argmax(z[j,:])        
        output += (X[m,j]*np.log(pis[q,l])+
                  (1-X[m,j])*np.log(1-pis[q,l]))
        
    return output 


@numba.jit(nopython=True,parallel=True)
def E_step_CEM(m,n,Q,X,z,alphas,pis,inds):
    Lq = np.log(alphas)
    for q in numba.prange(Q):
        for j in inds:
            if j!=m:
                l = np.argmax(z[j,:])                
                Lq[q] += (X[m,j]*np.log(pis[q,l])+\
                         (1-X[m,j])*np.log(1-pis[q,l]))
                
    return Lq

@numba.jit(nopython=True,parallel=True)
def M_step_lower_n_CEM(m,z,nq,nql,inds,X):
    q = np.argmax(z[m,:])
    nq[q] += 1

    ## nql
    for ind in numba.prange(len(inds)):
        j = inds[ind]
        l = np.argmax(z[j,:])
        if X[m,j]:
            nql[q,l] += 1
            nql[l,q] = nql[q,l]
            
@numba.jit(nopython=True,parallel=True)
def M_step_greater_n_CEM(m,z,nq,nql,inds,X,old_class,n):
    q = np.argmax(z[m,:])
    if old_class != q:
        nq[old_class] -= 1
        nq[q] += 1

        ## nql
        for ind in numba.prange(len(inds)):
            j = inds[ind]
            l = np.argmax(z[j,:])
            if X[m,j]:
                nql[q,l] += 1
                nql[l,q] = nql[q,l]

                nql[old_class,l] -= 1
                nql[l,old_class] = nql[old_class,l]


@numba.jit(nopython=True,parallel=True)
def M_step_pi_m_lower_n(m,z,Q,nq,nql,pis):
    q = np.argmax(z[m,:])
    
    ## pis
    for l in numba.prange(q):
        if nq[q]>0 and nq[l]>0:
            pis[q,l] = nql[q,l]/(nq[q]*nq[l])
            pis[l,q] = pis[q,l]
        else:
            pis[q,l] = eps
            pis[l,q] = eps

    if nq[q]>1:
        pis[q,q] = 2*nql[q,q]/(nq[q]*(nq[q]-1))
    else:
        pis[q,q] = eps

    return np.minimum(np.maximum(pis,eps),1-eps)


@numba.jit(nopython=True,parallel=True)
def M_step_pi_m_greater_n(m,z,Q,nq,nql,pis,old_class):
    q = np.argmax(z[m,:])
    pis = M_step_pi_m_lower_n(m,z,Q,nq,nql,pis)
    
    if old_class!=q:
        ## pis
        for l in numba.prange(old_class):
            if nq[old_class]>0 and nq[l]>0:
                pis[old_class,l] = nql[old_class,l]/(nq[old_class]*nq[l])
                pis[l,old_class] = pis[old_class,l]
            else:
                pis[old_class,l] = eps
                pis[l,old_class] = eps

        if nq[old_class]>1:
            pis[old_class,old_class] = 2*nql[old_class,old_class]/(nq[old_class]*(nq[old_class]-1))
        else:
            pis[old_class,old_class] = eps
        
    return pis


class OCEM():
    def __init__(self,Q):
        self.Q = Q
        
    def init_cem(self,inds):
        kmeans = KMeans(n_clusters = self.Q, embedding_method=GSVD(self.Q))
        labels_k = kmeans.fit_transform(self.X[np.ix_(inds,inds)])

        self.z = np.zeros(shape=(self.n,self.Q))
        self.z[inds] = np.eye(self.Q)[labels_k]

        self.nq = np.empty(shape=(self.Q,))
        self.nql = np.empty(shape=(self.Q,self.Q))
        
        ## nq
        for q in range(self.Q):
            self.nq[q] = len(labels_k[labels_k==q])
        
        ## nql
        for i in range(len(inds)):
            for j in range(i+1,len(inds)):
                if self.X[inds[i],inds[j]]:
                    self.nql[labels_k[i],labels_k[j]] += 1
                    self.nql[labels_k[j],labels_k[i]] = self.nql[labels_k[i],labels_k[j]]
                        
        self.alphas = np.maximum(self.nq/len(inds),eps)
        self.pis = np.empty(shape=(self.Q,self.Q))
        
        ## pi
        for q in range(self.Q):
            for l in range(q):
                if self.nq[q]>0 and self.nq[l]>0:
                    self.pis[q,l] = self.nql[q,l]/(self.nq[q]*self.nq[l])
                    self.pis[l,q] = self.pis[q,l]
                else:
                    self.pis[q,l] = eps
                    self.pis[l,q] = eps
                    
            if self.nq[q]>1:
                self.pis[q,q] = 2*self.nql[q,q]/(self.nq[q]*(self.nq[q]-1))
            else:
                self.pis[q,q] = eps
            
        self.pis = np.minimum(np.maximum(self.pis,eps),1-eps)
            
        
    def partial_log_likelihood(self,m,inds):
        inds2 = inds[:]
        if m>=self.n:
            m = m % self.n
            inds2.remove(m)

        return partial_log_likelihood(m,self.X@np.eye(self.n),np.copy(self.z),
                                      self.alphas,self.pis,np.array(inds2))
                
    
    def E_step(self,m):
        if m>=self.n:
            m = m % self.n
            self.old_class = np.argmax(self.z[m,:])
        self.Lq = E_step_CEM(m,self.n,self.Q,self.X@np.eye(self.n),\
                             self.z,self.alphas,self.pis,np.array(self.inds))
    
    
    def C_step(self,m):           
        if m>=self.n:
            m = m%self.n
            self.z[m,:] = 0
            
        q = np.argmax(self.Lq)
        self.z[m,q] = 1
            
    
    def M_step(self,m):
        if m<self.n:
            M_step_lower_n_CEM(m,self.z,self.nq,self.nql,np.array(self.inds),
                               self.X@np.eye(self.n))
            self.alphas = np.maximum(self.nq/(len(self.inds)+1),eps)
            self.pis = M_step_pi_m_lower_n(m,self.z,self.Q,self.nq,self.nql,self.pis)
        else:
            m = m%self.n
            M_step_greater_n_CEM(m,self.z,self.nq,self.nql,np.array(self.inds),
                                 self.X@np.eye(self.n),self.old_class,self.n)
            self.alphas = np.maximum(self.nq/self.n,eps)
            self.pis = M_step_pi_m_greater_n(m,self.z,self.Q,self.nq,self.nql,self.pis,self.old_class)
                     

    def fit(self,adjacency,inds0,N=1,print_likelihood=False):
        self.X = deepcopy(adjacency)
        self.n = self.X.shape[0]
        self.inds = list(deepcopy(inds0))
        
        self.init_cem(inds0)
        
        self.Lq = np.empty(shape=(self.Q,))
        
        n_iter = N*self.n
            
        for m in range(n_iter):
            if m>=self.n or m not in self.inds:
                if print_likelihood:
                    L1 = self.partial_log_likelihood(m,self.inds)
            
                self.E_step(m)
                self.C_step(m)
                
                if print_likelihood:
                    L3 = self.partial_log_likelihood(m,self.inds)

                self.M_step(m)
                
                if print_likelihood:
                    L2 = self.partial_log_likelihood(m,self.inds)

                    print(m,L1,L3,L2)

                    if L3<L1:
                        print("Error E-Step",L1-L3)                
                    if L2<L3:
                        print("Error M-Step",L3-L2,self.alphas[np.argmax(self.z[m,:])],np.argmax(self.z[m,:]))                
                    if L2<L1:
                        print(m,"Error",L1-L2)
                
            if m<self.n and m not in self.inds:                    
                self.inds.append(m)
                
        
        self.labels = np.argmax(self.z,axis=1)
        
        return self.labels