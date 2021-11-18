# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 09:22:05 2020

@author: Morten Østergaard Nielsen

This script contains a solution for the exercise from Lecture 6 in the 
Machine Learning course.
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier as MLPC
import time
import warnings

warnings.simplefilter(action='ignore') #Ignore warnings

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

#%%Multi Layer Perceptron (MLP) trained on PCA reduced data
hidden_units = 120      #Number of units/neurons in a hidden layer

n_components = 30
pca = PCA(n_components)

#PCA reduce the training and test set
trn_pca_set = pca.fit_transform(trn_set)
tst_pca_set = pca.transform(tst_set)

#Create a MLP classifier
mlpc = MLPC(hidden_units)

#Train the MLP and compute the training time.
start_time = time.time()
mlpc.fit(trn_pca_set, trn_targets)
fit_time = time.time() - start_time     #Compute the training/fiting time in seconds
print("Training Time: {:2.0f}m{:2.0f}s".format(fit_time//60, fit_time%60))

#Compute predictions and test accuracy.
pca_pred = mlpc.predict(tst_pca_set)
pca_acc = np.sum(pca_pred == tst_targets)/len(tst_targets) * 100
print("MLP Accuracy (PCA Data): {:.2f}".format(pca_acc))

#%%MLP trained on LDA reduced data
lda = LDA(n_components = 9)

#LDA reduce the training and test set
trn_lda_set = lda.fit_transform(trn_set, trn_targets)
tst_lda_set = lda.transform(tst_set)

#Create a MLP classifier
mlpc = MLPC(hidden_units)

#Train the MLP and compute the training time.
start_time = time.time()
mlpc.fit(trn_lda_set, trn_targets)
fit_time = time.time() - start_time     #Compute the training/fiting time in seconds
print("Training Time: {:2.0f}m{:2.0f}s".format(fit_time//60, fit_time%60))

#Compute predictions and test accuracy.
lda_pred = mlpc.predict(tst_lda_set)
lda_acc = np.sum(lda_pred == tst_targets)/len(tst_targets) * 100
print("MLP Accuracy (LDA Data): {:.2f}".format(lda_acc))

#%%Confusion matrix
#Compute the confusion matrices for both MLPs (trained on PCA and LDA)
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