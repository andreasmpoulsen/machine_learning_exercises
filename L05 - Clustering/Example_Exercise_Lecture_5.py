# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 09:22:05 2020

@author: Morten Østergaard Nielsen
"""

import numpy as np
from scipy.io import loadmat
from scipy.stats import multivariate_normal as norm
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM

#%%Load dataset
file = "data/2D568class.mat"
data = loadmat(file)

#Training sets:
trn5 = data["trn5_2dim"]/255
trn6 = data["trn6_2dim"]/255
trn8 = data["trn8_2dim"]/255

#Concatenate the training sets into one set:
trainset = np.concatenate([trn5, trn6, trn8])
np.random.shuffle(trainset)

#%%Gaussian Mixture Model (GMM)
#Fit GMM
num_kernels = 3
gmm = GMM(num_kernels)
gmm.fit(trainset)

#Estimate parameters for a bivariante Gaussian distribution for each class.
mean5 = np.mean(trn5, axis=0)
mean6 = np.mean(trn6, axis=0)
mean8 = np.mean(trn8, axis=0)

cov5 = np.cov(trn5.T)
cov6 = np.cov(trn6.T)
cov8 = np.cov(trn8.T)

# #Densities
l5 = norm(mean5, cov5)
l6 = norm(mean6, cov6)
l8 = norm(mean8, cov8)

#Create points to do a contour a plot
x = np.linspace(-8., 8., 101)
y = np.linspace(-7.2, 7.2, 101)
X, Y = np.meshgrid(x, y)
r = np.c_[X.ravel(), Y.ravel()]

#Get the individual means and covariances from the GMM to create seperate 
#densities. 
cov1, cov2, cov3 = gmm.covariances_
mean1, mean2, mean3 = gmm.means_
l1 = norm(mean1, cov1)
l2 = norm(mean2, cov2)
l3 = norm(mean3, cov3)

#Compute the probabilities to greate contours
Z = np.exp(gmm.score_samples(r).reshape(X.shape))

Z1 = l1.pdf(r).reshape(X.shape)
Z2 = l2.pdf(r).reshape(X.shape)
Z3 = l3.pdf(r).reshape(X.shape)

Z5 = l5.pdf(r).reshape(X.shape)
Z6 = l6.pdf(r).reshape(X.shape)
Z8 = l8.pdf(r).reshape(X.shape)

#Plot contours for the GMM, seperated GMM and individual estimated densities
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
cntr = ax1.contourf(X,Y,Z, levels = np.linspace(0.,0.045,11))
ax1.contour(X,Y,Z, colors = "k", levels = np.linspace(0.,0.045,11))
fig.colorbar(cntr, ax = ax1)
ax1.set_title("Gaussian Mixture Model (GMM)")

ax2.set_axisbelow(True) 
ax2.grid()
ax2.contour(X,Y,Z1, colors = "k")
ax2.contour(X,Y,Z2, colors = "r")
ax2.contour(X,Y,Z3, colors = "b")
ax2.set_title("Seperated GMM")

ax3.set_axisbelow(True) 
ax3.grid()
cntr1 = ax3.contour(X,Y,Z5, colors = "k")
cntr2 = ax3.contour(X,Y,Z6, colors = "r")
cntr3 = ax3.contour(X,Y,Z8, colors = "b")
h1, _ = cntr1.legend_elements()
h2, _ = cntr2.legend_elements()
h3, _ = cntr3.legend_elements()
ax3.legend([h1[0], h2[0], h3[0]], ["Class 5", "Class 6", "Class 8"])
ax3.set_title("Esitmated Densities for each class")

#%% Homemade EM-Algorithm
def _init_step(x, n_components = 1, init_mean = None):
    '''
    Computes initial values of the EM-algorithm

    Parameters
    ----------
    x : numpy.ndarray
        Data points from your training set.
    n_components : int, optional
        Number of Guassian kernels to include in the GMM. The default is 1.
    init_mean : numpy.ndarray, optional
        Initial value of the means. The default is None.

    Returns
    -------
    init_mean : numpy.ndarray
        Initial mean values.
    init_cov : numpy.ndarray
        Initial covariance matrices.
    init_prior : numpy.ndarray
        Initial priors.

    '''
    #If no initial means are given, choosen random points from data
    if init_mean is None:
        init_mean = x[np.random.choice(len(x), size= n_components, 
                                        replace = False)]
    
    N, d = x.shape      #Get the shape of the data.
    
    #Prepare empty arrays for init. covariances and priors.
    init_cov = np.zeros((n_components, d, d))
    init_prior = np.zeros(n_components)
    
    #Small k-means step to make labels for the clusters
    dis = np.linalg.norm(x[:,None] - init_mean, axis = -1)
    pred_lbl = np.argmin(dis, axis = -1)
    
    #For each Gauss kernel estimate covariance and prior from the labelled data.
    for i in range(n_components):
        init_cov[i] = np.cov(x[pred_lbl == i].T)
        init_prior[i] = np.sum(pred_lbl == i)/len(pred_lbl)
        # init_cov[i] = np.eye(d)
        # init_prior[i] = 1./n_components
        
    return init_mean, init_cov, init_prior    

def _log_likelihood(x, mean, cov, prior):
    lh = np.vstack([norm.pdf(x, mean[i], cov[i]) for i in range(len(prior))]).T
    return np.sum(np.log(np.sum(prior * lh, axis = -1)))

def _E_step(x, mean, cov, prior):
    #compute likelihoods for the data points belonging to each cluster
    lh = np.vstack([norm.pdf(x, mean[i], cov[i]) for i in range(len(prior))]).T
    
    c = np.sum(lh * prior, axis = -1).reshape(-1,1)
    gamma = np.exp(np.log(prior) + np.log(lh) - np.log(c))
    return gamma

def _M_step(x, gamma):
    Nk = gamma.sum(0)
    mean = np.dot(gamma.T, trainset)/Nk[:,None]
    
    y = x[:,None] - mean
    cov = np.sum(gamma[:,:,None, None] * np.einsum("nij, nik -> nijk", y,y), axis = 0)/Nk[:, None, None]
    
    prior = Nk/len(x)
    
    return mean, cov, prior

def EM_algorithm(x, n_components = 1, max_iter = 100, init_mean = None):
    '''
    The EM-algorithm. Choose a number of iterations for repeating the E- and
    M-step.

    Parameters
    ----------
    x : numpy.ndarray
        Data points from your training set.
    n_components : int, optional
        Number of Guassian kernels to include in the GMM. The default is 1.
    n_iter : int, optional
        Number of iterations. The default is 100.
    init_mean : numpy.ndarray, optional
        Initial value of the means. The default is None.

    Returns
    -------
    mean : numpy.ndarray
        Mean values.
    cov : numpy.ndarray
        Covariance matrices.
    prior : numpy.ndarray
        Priors.
    '''
    
    #Get initial parameters
    mean, cov, prior = _init_step(x, n_components, init_mean)
    
    llh = _log_likelihood(x, mean, cov, prior)
    stop = False
    k = 0
    
    while not(stop) and k < max_iter:
        #Predict new cluster labels with the E-step
        gamma = _E_step(x, mean, cov, prior)
        #Update mean, cov, and prior with the M-step using the new labels.
        new_mean, new_cov, new_prior = _M_step(x, gamma)
        
        new_llh = _log_likelihood(x, new_mean, new_cov, new_prior)
        
        if new_llh <= llh:
            stop = True
        else:
            llh = new_llh
            mean = new_mean
            cov = new_cov
            prior = new_prior
    return mean, cov, prior

def GaussMixModel(x, mean, cov, prior):
    '''
    Computes the posterior probability for the GMM.

    Parameters
    ----------
    x : numpy.ndarray
        Data points from a data set, e.g. test set.
    mean : numpy.ndarray
        Mean values.
    cov : numpy.ndarray
        Covariance matrices.
    prior : numpy.ndarray
        Priors.
        
    Returns
    -------
    p : numpy.ndarray
        posterior probabilities.

    '''
    p = 0.
    for i in range(len(mean)):
        p = p + prior[i] * norm.pdf(x, mean[i], cov[i])
    return p

def kmeans(x, k, n_iter = 100):
    '''
    Simple k-means method for computing means from k clusters.

    Parameters
    ----------
    x : numpy.ndarray
        Data points from your training set.
    k : int
        number of clusters.
    n_iter : int, optional
        Number of iterations. The default is 100.

    Returns
    -------
    mean : numpy.ndarray
        Mean values for each k-clusters.
    '''
    #Pick k random points as initial values
    mean = x[np.random.choice(len(x), size = k, replace = False)]
    
    #Repeat:
    for _ in range(n_iter):
        #Compute the distance to the mean of each cluster
        dis = np.linalg.norm(x[:,None] - mean, axis = -1)
        #Predict labels 
        pred_lbl = np.argmax(dis, axis = -1)
        
        #Update the cluster mean values from the new labels.
        for i in range(len(mean)):
            mean[i] = np.mean(x[pred_lbl==i], axis = 0)
    
    return mean

#Small test of the EM-algorithm
#If you pick a different inital mean vector, you might need more iterations for
#the algorithm to converge. Also, there is a change the EM-algorithm will
#diverge instead. The GMM class in scikit-learn uses k-means to find the 
#initial values but play around with different values here.
init_mean = kmeans(trainset, 3) 
mean, cov, prior = EM_algorithm(trainset, 3, 100, init_mean)
Z = GaussMixModel(r, mean, cov, prior).reshape(X.shape)
fig, ax = plt.subplots()
cntr = ax.contourf(X,Y,Z, levels = np.linspace(0.,0.05,11))
ax.contour(X,Y,Z, colors = "k", levels = np.linspace(0.,0.05,11))
fig.colorbar(cntr, ax = ax)
ax.set_title("GMM (Homemade EM-algorithm)")
