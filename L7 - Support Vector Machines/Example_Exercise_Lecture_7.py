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
from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time

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

#%%Support Vector Machine (SVM)
#Maximum number of iterations. Even if the SVM doesn't converge it will stop anyway.
n_iter = 1000

#Create SVM for classification (SVC). The standard kernel in the scikit-learn
#is the RBF (Gaussian Kernel).
svc = svm.SVC(max_iter = n_iter)

start_time = time.time()    #Time in seconds when starting the training/fiting.
svc.fit(trn_set, trn_targets)   #Train/fit the SVC
fit_time = time.time() - start_time     #Compute the training/fiting time in seconds
print("Training Time: {:2.0f}m{:2.0f}s".format(fit_time//60, fit_time%60))

#Predictions
pred = svc.predict(tst_set)

end_time = time.time() - start_time #Compute the total train and predict time in seconds
print("Total Running Time: {:2.0f}m{:2.0f}s".format(end_time//60, end_time%60))

#Compute accuracy
acc = np.sum(pred == tst_targets)/len(tst_targets) * 100
print("SVM Accuracy: {:.2f}".format(acc))

#%%Confusion matrix
#Compute the confusion matrix
cm = confusion_matrix(tst_targets, pred, normalize = "true")

#Prepare for plotting
cm = ConfusionMatrixDisplay(cm, classes)

#Plot Confusion matrices
fig, ax = plt.subplots()
cm.plot(cmap = "Blues", ax = ax)
ax.set_title("Confusion Matrix SVM")