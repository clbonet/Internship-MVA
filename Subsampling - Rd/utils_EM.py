import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


def multivariate_gaussian(pos, mu, Sigma):
    """
        Return the multivariate Gaussian distribution on array pos.

        Input:
        - pos is an array constructed by packing the meshed arrays of variables
        x_1, x_2, x_3, ..., x_k into its _last_ dimension.
        - mu: expectation of a gaussian
        - sigma: standard deviation of a gaussian
        
        Source: https://stackoverflow.com/questions/28342968/how-to-plot-a-2d-gaussian-with-different-sigma
    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N


def plot_ellipses(i,j,axes,K,mus,sigmas,alphas,X,dico=None):
    """
        plot ellipses which corresponds to the covariances matrix (mus) for each cluster, as well as
        the centers of each clusters (expectations mus) and the datas
        
        Input:
        - i: dimension in abscissa
        - j: dimension in ordored
        - axes: object to plot
        - K: number of clusters
        - mus: centers of clusters
        - sigmas: covariance matrix of each cluster
        - alphas: list of p_k (P(Zi=k)=p_k) 
        - dico: (default: None) dico of cem returning for each point its cluster
        - X: points
        
        If dico is None, in order to determine in which cluster each point belongs
        computes for each k: p(X,Z=k)=p(X|Z=k)p(Z=k) and keeping the max
        Else, use dico
    """

    xx = np.linspace(min(X[:,i]),max(X[:,i]),100)
    yy = np.linspace(min(X[:,j]),max(X[:,j]),100)

    XX, YY = np.meshgrid(xx, yy)

    pos = np.empty(XX.shape + (2,))
    pos[:, :, 0] = XX
    pos[:, :, 1] = YY
    
#     Z = [multivariate_gaussian(pos, mus[l], sigmas[l]) for l in range(K)]
#     Z = [multivariate_gaussian(pos, mus[l][[i,j]], sigmas[l][[i,j]][:,[i,j]]) for l in range(K)]
    
    Z = [multivariate_gaussian(pos, mus[l][[i,j]], sigmas[l][np.ix_([i,j],[i,j])]) for l in range(K)]
    
    Y_predicted = np.zeros(len(X))
    
    colors = ["blue","red","green","yellow"]
    for l in range(len(X)):
        if dico is not None:
            ind = dico[l]
        else:
            maxi = -np.inf
            ind = -1
            for k in range(K):
                proba = alphas[k]*st.multivariate_normal(mus[k], sigmas[k]).pdf(X[l])
                if(proba>maxi):
                    maxi = proba
                    ind = k
                    
        axes.scatter(X[l,i],X[l,j],color=colors[ind])
        Y_predicted[l] = ind
            
    for l in range(K):
        axes.scatter(mus[l][[i,j]][0],mus[l][[i,j]][1],marker="+",color="black",s=50)
        axes.contour(XX, YY, Z[l], colors='black')