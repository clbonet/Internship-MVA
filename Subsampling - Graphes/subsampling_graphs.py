import numpy as np

from dppy.finite_dpps import FiniteDPP

from utils_stats import *


def uniformSubsampling(adjacency,k):
    """
        draws uniformly k nodes
        
        Input:
        - adjacency: adjacency matrix
        - k: number of nodes to draw
        
        Output:
        - indices of the sample
    """
    inds = np.random.choice(adjacency.shape[0],k,replace=False)
    return inds


def uniformEdgeSubsampling(adjacency,k):
    """
        draws uniformly k edges and return the incident subgraph
        
        Input:
        - adjacency: adjacency matrix
        - k: number of edges to draw
        
        Output:
        - inds: indices of the sample
        - new_adjacency: incident adjacency matrix
    """
    adjacency_list = []
    for i in range(adjacency.shape[0]):
        for j in range(i+1,adjacency.shape[0]):
            if adjacency[i,j]:
                adjacency_list.append((i,j))
    
    ne = len(adjacency_list)
    inds_edges = np.random.choice(ne,k,replace=False)
    
    inds = set()
    for i in inds_edges:
        e = adjacency_list[i]
        inds.add(e[0])
        inds.add(e[1])
    inds = list(inds)
    
    new_adjacency = np.zeros(shape=adjacency.shape)

    for i in inds_edges:
        e = adjacency_list[i]
        new_adjacency[(e[0],e[1])] = 1
        new_adjacency[(e[1],e[0])] = 1
        
    return inds,new_adjacency


def starSampling(adjacency,k):
    """
        Neighborhood/Star sampling
        
        Input:
        - adjacency: adjacency matrix
        - k: number of nodes to draw
        
        Output:
        - indices of the sample
    """
    inds = np.random.choice(adjacency.shape[0],k,replace=False)

    inds2 = set(inds)
    for i in inds:
        for j in range(adjacency.shape[0]):
            if adjacency[i,j]:
                inds2.add(j)

    inds2 = list(inds2)
    
    return inds2


def DPP_graph(adjacency,p=10):
    """
        DPP on graph
        
        Reference: Graph Sampling with Determinantal Processes, Tremblay and Amblard and Barthelmé, 2017 : https://arxiv.org/pdf/1703.01594.pdf
        
        Input:
        - adjacency: adjacency matrix
        - p: proportion for the filter
        
        Output:
        - indices of the sample
    """
    n = adjacency.shape[0]
    D = adjacency@np.ones(n)
    L = D-adjacency
    
    eig_vals,eig_vecs = np.linalg.eigh(L)
    
    filt = np.diag(eig_vals<np.percentile(eig_vals,p))
    K = eig_vecs@filt@eig_vecs.T

    DPP = FiniteDPP('correlation',
                **{'K': K})
    DPP.sample_exact()
    
    return DPP.list_of_samples[-1]


def DPP_kernel(adjacency,kernel,k=50):
    """
        DPP on graph using the kernel as L
        
        Input:
        - adjacency: adjacency matrix
        - kernel: kernel for L
        - k: number of nodes
        
        Output:
        - indices of the sample
    """
    L = kernel(adjacency)
    DPP = FiniteDPP('likelihood',
                    **{'L': L})

    inds = DPP.sample_exact_k_dpp(k)
    return inds


def MHG(adjacency,k=50,nit=300,p=None,T0=1,gamma=1):
    """
        Metropolis Hasting for graphs 
        
        Reference: Metropolis Algorithms for Representative Subgraph Sampling, Hübler, Kriegel, Borgwards et Ghahramani, 2008 : http://mlcb.is.tuebingen.mpg.de/Veroeffentlichungen/papers/HueBorKriGha08.pdf
        
        Input; 
        - adjacency: adjacency matrix
        - k: number of nodes in the sample
        - nit: number of iterations
        - p: parameter for density (default None, if None, p=10*(k/n)*log_{10}(n)
        - T0: initial temperature for annealing
        - gamma: geometric reason for annealing
        
        Output:
        - Indices of the sample
    """
    n = adjacency.shape[0]
    
    if p is None:
        p = 10*(np.sum(adjacency@np.eye(n))/(2*n))*np.log(n)/np.log(10)
    
    S_c = np.random.choice(n,k,replace=False)
    S_b = np.copy(S_c)
    T = T0
    
    Cd1 = clustering_dist(adjacency)
    Cd_b = clustering_dist(adjacency[np.ix_(S_b,S_b)],len(Cd1)-1)
    d_b = np.sum(np.abs(Cd1-Cd_b))/len(Cd1)
    
    for i in range(nit):
        v = np.random.choice(S_c)
        
        S2 = np.ones((n,),dtype=int)
        S2[S_c] = 0
        S2[v] = 1
        
        w = np.random.choice(np.nonzero(adjacency[S2])[0])
        
        S_n = np.copy(S_c)
        S_n[np.where(S_c==v)[0][0]] = w
        
        alpha = np.random.rand()
        
        Cd1 = clustering_dist(adjacency)
        Cd_c = clustering_dist(adjacency[np.ix_(S_c,S_c)],len(Cd1)-1)
        Cd_n = clustering_dist(adjacency[np.ix_(S_n,S_n)],len(Cd1)-1)
        
        d_c = np.sum(np.abs(Cd1-Cd_c))/len(Cd1)
        d_n = np.sum(np.abs(Cd1-Cd_n))/len(Cd1)
        
        if alpha<(d_c/d_n)**(p/T):
            S_c = np.copy(S_n)
            if d_c<d_b:
                S_b = np.copy(S_c)
                d_b = d_c
                
        T *= gamma
        
    return S_b


def XMC(adjacency,k=50,nit=300,p=None,T0=1,gamma=1):
    """
        MCMC Expansion Sampler (XMC)
        
        Reference: - Sampling Community Structure, Maiya and Berger-Wolf, 2010 : http://arun.maiya.net/papers/maiya_etal-sampcomm.pdf
                   - Metropolis Algorithms for Representative Subgraph Sampling, Hübler, Kriegel, Borgwards et Ghahramani, 2008 : http://mlcb.is.tuebingen.mpg.de/Veroeffentlichungen/papers/HueBorKriGha08.pdf
                   
        Input:
        - adjacency: adjacency matrix
        - k: number of nodes in the sample
        
        Output:
        - Indices of the sample
    """
    n = adjacency.shape[0]
    
    if p is None:
        p = 10*(np.sum(adjacency@np.eye(n))/(2*n))*np.log(n)/np.log(10)
    
    S_c = np.random.choice(n,k,replace=False)
    S_b = np.copy(S_c)
    T = T0
    
    ## dist
    neighbours = np.zeros((n,))
    for s in S_b:
        neighbours += adjacency[s,:]
    neighbours[S_b] = 0
    neighbours[neighbours>0] = 1

    cardNS = np.sum(neighbours)
    cardVS = n-k
    
    d_b = cardNS/cardVS
    
    for i in range(nit):
        v = np.random.choice(S_c)
        
        S2 = np.ones((n,),dtype=int)
        S2[S_c] = 0
        S2[v] = 1
        
        w = np.random.choice(np.nonzero(adjacency[S2])[0])
        
        S_n = np.copy(S_c)
        S_n[np.where(S_c==v)[0][0]] = w
        
        alpha = np.random.rand()
        

        ## dist S_n
        neighbours = np.zeros((n,))
        for s in S_n:
            neighbours += adjacency[s,:]
        neighbours[S_n] = 0
        neighbours[neighbours>0] = 1
        
        cardNS = np.sum(neighbours)
        cardVS = n-k
                
        d_n = cardNS/cardVS
        
        ## dist S_c
        neighbours = np.zeros((n,))
        for s in S_c:
            neighbours += adjacency[s,:]
        neighbours[S_c] = 0
        neighbours[neighbours>0] = 1
        
        cardNS = np.sum(neighbours)
        
        d_c = cardNS/cardVS
        
        if alpha<(d_c/d_n)**(p/T):
            S_c = np.copy(S_n)
            if d_c<d_b:
                S_b = np.copy(S_c)
                d_b = d_c
                
        T *= gamma
        
    return S_b


@numba.njit
def belong(i,inds):
    for j in inds:
        if i==j:
            return True
    return False


@numba.njit
def computeMax(n,adjacency,neighbours,S):
    cpt = 0
    vMax = 0
    for v in range(n):
        if neighbours[v]>0 and not belong(v,S):
            card = np.sum(adjacency[v,:]*(1-neighbours))
            if card>cpt:
                cpt = card
                vMax = v

    return vMax


def XSN(adjacency,k=100):
    """
        Snowball Expansion Sampler
        
        Reference: Sampling Community Structure, Maiya and Berger-Wolf, 2010 : http://arun.maiya.net/papers/maiya_etal-sampcomm.pdf
        
        Input:
        - adjacency: adjacency matrix
        - k: number of nodes in the sample
        
        Output:
        - Indices of the sample
    """
    n = adjacency.shape[0]
    
    v = np.random.choice(n)
    S = [v]
    
    while len(S)<k:
        neighbours = np.zeros((n,))
        for s in S:
            neighbours += adjacency[s,:]
        neighbours[S] = 1
        neighbours[neighbours>0] = 1
        
        vMax = computeMax(n,adjacency,neighbours,np.array(S))
        
        S.append(vMax)
    
    return S


def RW(adjacency,k,c=0.15):
    """
        Random Walk with Random Jump
        
        Input:
        - adjacency: adjacency matrix
        - k: number of nodes
        - c: parameter for random jump (c=0.15 by default)
        
        Output:
        - Indices of the sample
    """
    n = adjacency.shape[0]    
    deg = adjacency@np.ones(n)
    
    ind0 = np.random.randint(n)
    inds = [ind0]
    
    while len(inds)<k:
        U = np.random.rand()
        if U<c:
            ind = np.random.randint(n)
        else:
            ind = np.random.choice(n,p=adjacency[inds[-1],:]/deg[inds[-1]])
            
        if ind not in inds:
            inds.append(ind)
        
    return inds
