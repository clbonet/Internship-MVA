import numpy as np

from networkx.generators.community import stochastic_block_model
from networkx.convert_matrix import to_scipy_sparse_matrix


def generates_equal_sbm(n_clusters,n_by_cluster,p_in,p_out):

    p = np.ones((n_clusters,n_clusters))*p_out
    for i in range(n_clusters):
        p[i,i] = p_in

    graph_sbm = stochastic_block_model([n_by_cluster for k in range(n_clusters)],p)
    adjacency_sbm = to_scipy_sparse_matrix(graph_sbm)
    labels_sbm = np.ones(shape=(n_by_cluster*n_clusters,),dtype=int)

    for k in range(n_clusters):
        labels_sbm[k*n_by_cluster:(k+1)*n_by_cluster] *= k
        
    return adjacency_sbm,labels_sbm

def get_p(n,p_in,p_out):
    ## Affiliation Model
    p = np.ones((n,n))*p_out
    np.fill_diagonal(p,p_in)
    return p

def get_random_p(n,a,b):
    M = a*np.random.rand(n,n)
    p = (M+M.T)/2
    diag = (b-a)*np.random.rand(n)+a
    np.fill_diagonal(p,diag)
    return p

def get_random_p_inv(n,a,b):
    M = (b-a)*np.random.rand(n,n)+a
    p = (M+M.T)/2
    diag = a*np.random.rand(n)
    np.fill_diagonal(p,diag)
    return p

def create_graph(pi,nq):
    graph = stochastic_block_model(nq,pi)
    adjacency = to_scipy_sparse_matrix(graph)
    labels = np.ones(np.sum(nq),dtype=int)
    
    nq_sum = np.cumsum(nq)
    labels[:nq[0]] *= 0
    for q in range(1,len(nq)-1):
        labels[nq_sum[q-1]:nq_sum[q]] *= q
    labels[nq_sum[-2]:] *= len(nq)-1
    
    return adjacency,labels    
