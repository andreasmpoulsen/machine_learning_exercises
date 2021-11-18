# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 09:22:05 2020

@author: Morten Østergaard Nielsen

This script contains a solution for the exercise from Lecture 6 in the 
Machine Learning course.
"""

import numpy as np
from scipy.io import loadmat
from scipy.stats import multivariate_normal as norm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#%%Functions
def create_comp_datasets(data_dict):
    '''
    Function for creating complete training and test sets containing
    all classes.
    '''
    #Empty list
    trainset = []
    traintargets =[]
    testset = []
    testtargets =[]
    
    #For each class
    for i in range(10):
        trainset.append(data_dict["train%d"%i])
        traintargets.append(np.full(len(data_dict["train%d"%i]),i))
        testset.append(data_dict["test%d"%i])
        testtargets.append(np.full(len(data_dict["test%d"%i]),i))
    
    #Concatenate into to complete datasets
    trainset = np.concatenate(trainset)
    traintargets = np.concatenate(traintargets)
    testset = np.concatenate(testset)
    testtargets = np.concatenate(testtargets)
    return trainset, traintargets, testset, testtargets

def est_params(trn_set, trn_targets):
    '''
    Function for estimating the parameters for multiple gaussian distributions.

    Parameters
    ----------
    trn_set : numpy.ndarray
        Training set.
    trn_targets : numpy.ndarray
        Training targets / class labels.

    Returns
    -------
    means : numpy.ndarray
        Mean vectors for each class.
    covs : numpy.ndarray
        Covariance matrices for each class.
    '''
    #Get the data dimension
    _, d = trn_set.shape 
    
    #Zero arrays for storing means and covs.
    means = np.zeros((10,d))
    covs = np.zeros((10, d, d))
    
    #For each class compute mean and cov.
    for i in range(10):
        indx = trn_targets == i
        means[i] = np.mean(trn_set[indx], axis = 0)
        covs[i] = np.cov(trn_set[indx].T)
    return means, covs

def predict(tst_set, means, covs):
    '''
    Function for making the class prediction based on maximum likelihood.

    Parameters
    ----------
    tst_set : numpy.ndarray
        Test set.
    means : numpy.ndarray
        Mean vectors for each class.
    covs : numpy.ndarray
        Covariance matrices for each class.
        
    Returns
    -------
    preds : numpy.ndarray
        Class predictions.
    '''
    probs = []
    for i in range(len(covs)):
        probs.append(norm.pdf(tst_set, means[i], covs[i]))
    probs = np.c_[tuple(probs)]
    preds = np.argmax(probs, axis = 1)
    return preds

#%%Load dataset
file = "data/mnist_all.mat"
data = loadmat(file)

#Complete training and test sets
trn_set, trn_targets, tst_set, tst_targets = create_comp_datasets(data)

#Normalize datasets:
#Each image is in 8-bit intergers, so the max value is 2^8 - 1 = 255.
#By normalizing the images, each pixel is now a floating point value 
#between 0 and 1. This doesn't change the result in the end but it is just good 
#practise.
trn_set = trn_set/255
tst_set = tst_set/255

#A list of the class names.
classes = np.arange(10)

#%%Dimentional Reduction: PCA and LDA
n_components = 9

#PCA
pca = PCA(n_components = n_components)
trn_pca_set = pca.fit_transform(trn_set)
tst_pca_set = pca.transform(tst_set)

#LDA
lda = LDA(n_components = n_components)
trn_lda_set = lda.fit_transform(trn_set, trn_targets)
tst_lda_set = lda.transform(tst_set)

#Proportion of Variance
pov_pca = np.sum(pca.explained_variance_ratio_)
pov_lda = np.sum(lda.explained_variance_ratio_)

if n_components == 2:
    #If the number of components is 2, we can plot the reduced data.
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)
    ax1.grid()
    ax2.grid()
    ax1.set_title("PCA (PoV: {:.5f})".format(pov_pca))
    ax2.set_title("LDA (PoV: {:.5f})".format(pov_lda))
    
    for i in range(10):
        indx = trn_targets == i
        ax1.scatter(trn_pca_set[indx,0], trn_pca_set[indx,1], 
                    marker = ".", label = "Class %d"%i)
        ax2.scatter(trn_lda_set[indx,0], trn_lda_set[indx,1], 
                    marker = ".", label = "Class %d"%i)
    ax1.legend()
    ax2.legend()
else:
    #Else, just print the PoV
    print("PoV (PCA): {:.5f}".format(pov_pca))
    print("PoV (LDA): {:.5f}".format(pov_lda))
    
#%%Classification
#Compute the parameters for each PCA and LDA reduced data.
pca_means, pca_covs = est_params(trn_pca_set, trn_targets)
lda_means, lda_covs = est_params(trn_lda_set, trn_targets)

#Compute predictions
pca_pred = predict(tst_pca_set, pca_means, pca_covs)
lda_pred = predict(tst_lda_set, lda_means, lda_covs)

#Compute accuracy
pca_acc = np.sum(pca_pred == tst_targets)/len(tst_targets) * 100
lda_acc = np.sum(lda_pred == tst_targets)/len(tst_targets) * 100

print("PCA Accuracy: {:.2f}".format(pca_acc))
print("LDA Accuracy: {:.2f}".format(lda_acc))

#%%Confusion matrix
#Compute the confusion matrices for PCA and LDA
pca_cm = confusion_matrix(tst_targets, pca_pred, normalize = "true")
lda_cm = confusion_matrix(tst_targets, lda_pred, normalize = "true")

#Prepare for plotting
pca_cm = ConfusionMatrixDisplay(pca_cm, classes)
lda_cm = ConfusionMatrixDisplay(lda_cm, classes)

#Plot Confusion matrices
fig, (ax1, ax2) = plt.subplots(1,2)
pca_cm.plot(cmap = "Blues", ax = ax1)
lda_cm.plot(cmap = "Blues", ax = ax2)
ax1.set_title("Confusion Matrix PCA")
ax2.set_title("Confusion Matrix LDA")