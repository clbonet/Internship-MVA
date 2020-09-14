import numpy as np
import matplotlib.pyplot as plt

import numba

from scipy.stats import ks_2samp
from sknetwork.path.metrics import diameter
from sknetwork.clustering import modularity


@numba.jit
def clustering_coeff_full(A):
    degrees = A@np.ones(shape=(A.shape[0],))
    
    num = 0
    denom = 0
    
    for u in range(A.shape[0]):
        for v in range(A.shape[0]):
            for w in range(v+1,A.shape[0]):
                num += A[u,v]*A[v,w]*A[u,w]
        denom += degrees[u]*(degrees[u]-1)/2

    return num/denom


@numba.jit
def clustering_coeff(A,nodes=None):
    if nodes is None:
        nodes = range(A.shape[0])
        
    degrees = A@np.ones(shape=(A.shape[0],))
    
    output = np.zeros(shape=(len(nodes),))
    
    for v in nodes:
        d = degrees[v]*(degrees[v]-1)/2
        
        for u in range(len(A)):
            for w in range(u+1,len(A)):
                if A[u,v] and A[v,w] and A[u,w]:
                    output[v] += 1/d
                    
    return output


def clustering_dist(A,max_deg=None):
    Cv = clustering_coeff(A@np.eye(A.shape[0]))
    degrees = A@np.ones(shape=(A.shape[0],))
    
    if max_deg is None:
        max_deg = degrees.max()
    else:
        max_deg = max(degrees.max(),max_deg)
        
    output = np.zeros(shape=(int(max_deg)+1,))
    
    for d in range(int(max_deg)+1):
        filt = degrees==d
        if len(degrees[filt])>0:
            output[d] = np.mean(Cv[filt])
        
    return output


def cc_dist(adjacency1,adjacency2):
    """
        Clustering Coefficient Distribution based distance
    """
    Cd1 = clustering_dist(adjacency1)
    Cd2 = clustering_dist(adjacency2,len(Cd1)-1)
    return np.sum(np.abs(Cd1-Cd2))/len(Cd1)


def degree_dist(adjacency1,adjacency2):
    """
        Degree Distribution based distance (Kolmogorov-Smirnov)
    """
    degree1 = adjacency1@np.ones(adjacency1.shape[1])
    degree2 = adjacency2@np.ones(adjacency2.shape[1])

    degree1= (degree1-degree1.mean())/np.std(degree1)
    if np.std(degree2)>0:
        degree2 = (degree2-degree2.mean())/np.std(degree2)
    else:
        degree2 = (degree2-degree2.mean())

    degree1 = np.sort(degree1)
    degree2 = np.sort(degree2)
    deg_all = np.concatenate([degree1,degree2])

    cdf1 = np.searchsorted(degree1, deg_all, side='right') / len(degree1)
    cdf2 = np.searchsorted(degree2, deg_all, side='right') / len(degree2)

    return np.max(np.abs(cdf1-cdf2))


def diameter_dist(adjacency1,adjacency2):
    """
        Diameter based distance (percent deviation)
    """
    diameter1 = diameter(adjacency1)
    diameter2 = diameter(adjacency2)
    return np.abs((diameter1-diameter2)/diameter1)


def modularity_dist(adjacency1,labels1,adjacency2,labels2):
    """
        Modularity based distance (percent-deviation)
    """
    m1 = modularity(adjacency1,labels1)
    m2 = modularity(adjacency2,labels2)

    return np.abs((m1-m2)/m1)


def fraction_communities(labels1,labels2):
    """
        Fraction of communities kept
    """
    cpt = 0
    n = np.max(labels1)
    for k in range(np.min(labels1),n):
        cpt += len(labels2[labels2==k])/(n*len(labels1[labels1==k]))
        
    return cpt


def density(adjacency):
    m = np.sum(adjacency)
    n = adjacency.shape[0]
    return m/(n*(n-1))


def density_dist(adjacency1,adjacency2):
    d1 = density(adjacency1)
    d2 = density(adjacency2)
    return np.abs((d1-d2)/d1)


def computeC(adjacency,inds):
    neighbours = np.zeros((adjacency.shape[0],))
    for s in inds:
        neighbours += adjacency[s,:]
    neighbours[inds] = 0
    neighbours[neighbours>0] = 1
    return np.sum(neighbours)


def separability(adjacency,labels):
    outs = []
    for q in set(labels):
        inds = np.nonzero(labels==q)[0]
        mS = np.sum(adjacency[np.ix_(inds,inds)])/2
        c = computeC(adjacency@np.eye(adjacency.shape[1]),inds)
        if c>0:
            outs.append(mS/c)
    if len(outs)>0:
        return np.mean(outs)
    else:
        return 0


def separability_dist(adjacency1,labels1,adjacency2,labels2):
    d1 = separability(adjacency1,labels1)
    d2 = separability(adjacency2,labels2)
    return np.abs((d1-d2)/d1)


def conductance(adjacency,labels):
    outs = []
    for q in set(labels):
        inds = np.nonzero(labels==q)[0]
        mS = np.sum(adjacency[np.ix_(inds,inds)])/2
        c = computeC(adjacency@np.eye(adjacency.shape[1]),inds)
        if 2*mS+c>0:
            outs.append(c/(2*mS+c))
    return np.mean(outs)


def conductance_dist(adjacency1,labels1,adjacency2,labels2):
    d1 = conductance(adjacency1,labels1)
    d2 = conductance(adjacency2,labels2)
    return np.abs((d1-d2)/d1)


def expansion(adjacency,labels):
    outs = []
    for q in set(labels):
        inds = np.nonzero(labels==q)[0]
        c = computeC(adjacency@np.eye(adjacency.shape[1]),inds)
        outs.append(len(inds)/c)

    return outs


def expansion_dist(adjacency1,labels1,adjacency2,labels2):
#     d1 = expansion(adjacency1,labels1).mean()
#     d2 = expansion(adjacency2,labels2).mean()
#     print("Exp",d1,d2)    
#     return np.abs((d1-d2)/d1)

    e1 = np.array(expansion(adjacency1,labels1))
    e2 = np.array(expansion(adjacency2,labels2))
    
    e1= (e1-e1.mean())/np.std(e1)
    e2 = (e2-e2.mean())/np.std(e2)

    e1 = np.sort(e1)
    e2 = np.sort(e2)
    e_all = np.concatenate([e1,e2])

    cdf1 = np.searchsorted(e1, e_all, side='right') / len(e1)
    cdf2 = np.searchsorted(e2, e_all, side='right') / len(e2)

    return np.max(np.abs(cdf1-cdf2))



def returnStats(adjacency1,inds,labels1=None,print_stats=False):
    adjacency2 = adjacency1[np.ix_(inds,inds)]
    if labels1 is not None:
        labels2 = labels1[inds]
    
    cc = cc_dist(adjacency1,adjacency2)
    deg = degree_dist(adjacency1,adjacency2)
    diam = diameter_dist(adjacency1,adjacency2)
    if labels1 is not None:
        mod = modularity_dist(adjacency1,labels1,adjacency2,labels2)
        frac = fraction_communities(labels1,labels2)
    
    if print_stats:
        print("Clustering Coefficient",cc)
        print("Degree",deg)
        print("Diameter",diam)
        if labels1 is not None:
            print("Modularity",mod)
            print("Fraction Communities",frac)
            
            for i in range(np.min(labels1),np.max(labels1)):
                print(i,len(labels2[labels2==i]),len(labels1[labels1==i]))
        
    if labels1 is not None:
        return cc,deg,diam,mod,frac
    else:
        return cc,deg,diam
    
    
def returnStats(adjacency1,inds,labels1=None,print_stats=False):
    adjacency2 = adjacency1[np.ix_(inds,inds)]
    if labels1 is not None:
        labels2 = labels1[inds]
    
    cc = cc_dist(adjacency1,adjacency2)
    deg = degree_dist(adjacency1,adjacency2)
    diam = diameter_dist(adjacency1,adjacency2)
    dens = density_dist(adjacency1,adjacency2)
    if labels1 is not None:
        mod = modularity_dist(adjacency1,labels1,adjacency2,labels2)
        frac = fraction_communities(labels1,labels2)
        sep = separability_dist(adjacency1,labels1,adjacency2,labels2)
        cond = conductance_dist(adjacency1,labels1,adjacency2,labels2)
    
    if print_stats:
        print("Clustering Coefficient",cc)
        print("Degree",deg)
        print("Diameter",diam)
        print("Density",dens)
        if labels1 is not None:
            print("Modularity",mod)
            print("Fraction Communities",frac)
            print("Separability",sep)
            print("Conductance",cond)
            
            for i in range(np.min(labels1),np.max(labels1)+1):
                print(i,len(labels2[labels2==i]),len(labels1[labels1==i]))
        
    if labels1 is not None:
        return cc,deg,diam,dens,mod,frac,sep,cond
    else:
        return cc,deg,diam,dens


def plotStats(adjacency,labels,indices_subsampling,plot_degree):
    inds = indices_subsampling
    print("n",adjacency.shape[0],"n_samples",len(inds))
    
    if labels is not None:
        print()
        for i in range(3):
            print(i,len(labels[inds][labels[inds]==i]),len(labels[labels==i]))
        
        
        nql = np.zeros((3,3))

        for ind_i,i in enumerate(inds):
            for ind_j in range(ind_i,len(inds)):
                j = inds[ind_j]
                if adjacency[i,j]:
                    nql[labels[inds][ind_i],labels[inds][ind_j]] += 1
                    if labels[inds][ind_i] !=  labels[inds][ind_j]:
                        nql[labels[inds][ind_j],labels[inds][ind_i]] += 1

        nql_full = np.zeros((3,3))

        for i in range(adjacency.shape[0]):
            for j in range(i,adjacency.shape[0]):
                if adjacency[i,j]:
                    nql_full[labels[i],labels[j]] += 1
                    if labels[i]!=labels[j]:
                        nql_full[labels[j],labels[i]] += 1

        print()

        print("proportion edges sub\n",nql/np.sum(nql))
        print()
        print("proportion edges full\n",nql_full/np.sum(nql_full))
        
    print()

    ## (see Metropolis article)
    Cd1 = clustering_dist(adjacency)
    Cd2 = clustering_dist(adjacency[np.ix_(inds,inds)],len(Cd1)-1)
    print("Average Difference Clustering Coefficient",np.sum(np.abs(Cd1-Cd2))/len(Cd1))
    
    print()
    if plot_degree:
        degree = adjacency@np.ones(adjacency.shape[0])
        degree_sub = adjacency[np.ix_(inds,inds)]@np.ones(len(inds))
        
        degree = (degree-degree.mean())/np.std(degree)
        degree_sub = (degree_sub-degree_sub.mean())/np.std(degree_sub)
        
        fig,ax = plt.subplots(1,2,figsize=(15,7))
        ax[0].hist(degree,bins=20)
        ax[0].set_title("Full data")
        ax[1].hist(degree_sub,bins=20)
        ax[1].set_title("Subsampling")
        fig.suptitle("Degrees Distributions",fontsize=16)
        
        print("D-stat:",ks_2samp(degree,degree_sub))
        
        degree = np.sort(degree)
        degree_sub = np.sort(degree_sub)
        deg_all = np.concatenate([degree,degree_sub])
        
        cdf1 = np.searchsorted(degree, deg_all, side='right') / len(degree)
        cdf2 = np.searchsorted(degree_sub, deg_all, side='right') / len(degree_sub)
        
        print("D-stat - score:",np.max(np.abs(cdf1-cdf2)))
        
    diameter_sub = diameter(adjacency[np.ix_(inds,inds)])
    diameter_full = diameter(adjacency)
    print("Diameter",diameter_sub,diameter_full)
    print("Diameter Percent Deviation",(diameter_sub-diameter_full)/diameter_full)
#     print("Diameter Percent Deviation",(diameter_full-diameter_sub)/diameter_sub)
