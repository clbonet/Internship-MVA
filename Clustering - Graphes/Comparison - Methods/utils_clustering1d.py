import numpy as np
import matplotlib.pyplot as plt

def plot_score(f,labels,ax=plt):    
    y = np.ones(f.shape)
    
    colors = ["blue","red","green","yellow","grey","purple","cyan","lime","black"]
    scale = [1000,800,600,500,400,300,200,100,50]
    
    k = np.max(labels)
    
    for i in range(min(k+1,9)):
        ax.scatter(f[labels==i],y[labels==i],s=scale[i],color=colors[i],label="cluster "+str(i))
    
    if ax == plt:
        ax.title("scores")
        ax.yticks([])
    else:
        ax.set_title("scores")
        ax.set_yticks([])
        

def plot_iterations(fs,labels,inds=None):
    if inds is None:
        inds = range(len(fs))
    
    fig,ax = plt.subplots(len(fs)//2,2,figsize=(15,7))
    
    for i in range(len(fs)//2):
        plot_score(fs[2*i],labels,ax[i,0])
#         ax[i,0].set_title("$f_{"+str(2*i)+"}$")
        ax[i,0].set_title("$f_{"+str(inds[2*i])+"}$")
        plot_score(fs[2*i+1],labels,ax[i,1])
#         ax[i,1].set_title("$f_{"+str(2*i+1)+"}$")
        ax[i,1].set_title("$f_{"+str(inds[2*i+1])+"}$")
        
        
def plot_score2d(f,labels,ax=plt):
    for k in range(np.max(labels)+1):
        ax.scatter(f[labels==k][:,0],f[labels==k][:,1])
        
        
def plot_iterations2d(fs,labels,inds=None):
    if inds is None:
        inds = range(len(fs))
        
    fig,ax = plt.subplots(len(fs)//2,2,figsize=(10,10))

    for i in range(len(fs)//2):
        plot_score2d(fs[2*i],labels,ax[i,0])
        ax[i,0].set_title("$f_{"+str(inds[2*i])+"}$")
        plot_score2d(fs[2*i+1],labels,ax[i,1])
        ax[i,1].set_title("$f_{"+str(inds[2*i+1])+"}$")
