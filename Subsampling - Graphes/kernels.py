import numpy as np
import numba

from sklearn.metrics.pairwise import euclidean_distances
from scipy.linalg import expm
from sknetwork.path import distance


def kernel_adj_dist(adjacency):
    """
        Input: adjacency matrix
        Output: matrix of e^(-\|A_i-A_j\|^2/2)
    """
    dist = euclidean_distances(adjacency,adjacency) # # of different neighbours
    return np.exp(-dist**2/2)


def kernel_Laplacian(adjacency):
    """
        Input: adjacency matrix
        Output: e^(-L/2) with L the Laplacian
    """
    D = adjacency@np.ones(adjacency.shape[1])
    L = np.diag(D)-adjacency
    return expm(-L/2)


def kernel_Generalized_Laplacian(adjacency):
    """
        Input: adjacency matrix
        Output: e^(-\tilde{L}/2) with \tilde{L}=D^{-1/2}LD^{-1/2} the generalized Laplacian
    """
    D = adjacency@np.ones(adjacency.shape[1])
    D_sqrt_inv = np.linalg.inv(np.diag(D**(1/2)))
    L = np.eye(D.shape[0])-D_sqrt_inv@adjacency@D_sqrt_inv ## Normalized Laplacian
    return expm(-L/2)


def kernel_dist(adjacency):
    """
        Input: adjacency matrix
        Output: matrix of e^(-\|u-v\|^2/2) with \|.\| equal to the length of the shortest path 
        between u and v
    """
    dist = distance(adjacency)
    return np.exp(-dist**2/2)


def regularizedLaplacian(adjacency):
    """
        Input: adjacency matrix
        Output: (I+L)^-1 the regularized Laplacian
    """
    D = adjacency@np.ones(adjacency.shape[1])
    L = np.diag(D)-adjacency
    return np.linalg.inv(L+np.eye(adjacency.shape[0]))