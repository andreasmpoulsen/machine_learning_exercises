# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 09:22:05 2020

@author: Morten Østergaard Nielsen

This script contains two solutions for the exercise from Lecture 4 in the 
Machine Learning course. The first solution uses Principal Component Analysis 
(PCA) and the other uses Linear Discriminant Analysis (LDA).
"""

import numpy as np
from scipy.io import loadmat
from scipy.stats import multivariate_normal as norm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# %%Load dataset
file = "Datasets/mnist_all.mat"
data = loadmat(file)

# Training sets:
# Each image is in 8-bit intergers, so the max value is 2^8 - 1 = 255.
# By normalizing the images, each pixel is now a floating point value
# between 0 and 1. This doesn't change the result in the end but it is just good
# practise.
train5 = data["train5"]/255
train6 = data["train6"]/255
train8 = data["train8"]/255

# Create targets/classes for the training set
trn_target5 = 5*np.ones(len(train5))
trn_target6 = 6*np.ones(len(train6))
trn_target8 = 8*np.ones(len(train8))

# Concatenate the training sets into one set:
trainset = np.concatenate([train5, train6, train8])
trn_targets = np.concatenate([trn_target5, trn_target6, trn_target8])

# Test sets.
test5 = data["test5"]/255
test6 = data["test6"]/255
test8 = data["test8"]/255

# Create targets/classes for the test set
tst_target5 = 5*np.ones(len(test5))
tst_target6 = 6*np.ones(len(test6))
tst_target8 = 8*np.ones(len(test8))

# Concatenate the test sets into one set:
testset = np.concatenate([test5, test6, test8])
tst_targets = np.concatenate([tst_target5, tst_target6, tst_target8])

# A list of the class names.
classes = np.array([5, 6, 8])

# %%Part 1: Lower-Dimensional Reduction (PCA)
# The PCA class in scikit-learn fits a covariance matrix and compute
# eigenvectors for you. PCA doesn't assume any knowledge about the classes, so
# you have to use the concatenated training set
pca = PCA(n_components=2)
pca.fit(trainset)

# Lower-dimensional reduction of the training sets.
x5 = pca.transform(train5)
x6 = pca.transform(train6)
x8 = pca.transform(train8)

# Scatter plot of the dimensional-reduced data
fig, ax = plt.subplots(figsize=(12, 8))
fig.suptitle("PCA")
ax.set_axisbelow(True)
ax.grid()
ax.scatter(x5[:, 0], x5[:, 1], c='k', marker="x", label="Class 5")
ax.scatter(x6[:, 0], x6[:, 1], c='r', marker="x", label="Class 6")
ax.scatter(x8[:, 0], x8[:, 1], c='b', marker="x", label="Class 8")
ax.legend()

# Lower-dimensional reduction of the training sets.
tst = pca.transform(testset)

# %%Part 2: Classification (PCA)
# Estimate parameters for a bivariante Gaussian distribution.
mean5 = np.mean(x5, axis=0)
mean6 = np.mean(x6, axis=0)
mean8 = np.mean(x8, axis=0)

cov5 = np.cov(x5.T)
cov6 = np.cov(x6.T)
cov8 = np.cov(x8.T)

# Densities
l5 = norm(mean5, cov5)
l6 = norm(mean6, cov6)
l8 = norm(mean8, cov8)

# Compute a posterior probabilities
p5 = l5.pdf(tst)
p6 = l6.pdf(tst)
p8 = l8.pdf(tst)

# Compute predictions
pred = np.argmax(np.c_[p5, p6, p8], axis=1)
pred = classes[pred]

# Compute accuracy
acc5 = np.sum(pred[tst_targets == 5] == 5)/len(tst_target5) * 100
acc6 = np.sum(pred[tst_targets == 6] == 6)/len(tst_target6) * 100
acc8 = np.sum(pred[tst_targets == 8] == 8)/len(tst_target8) * 100
acc = np.sum(pred == tst_targets)/len(tst_targets) * 100
print("-"*10 + "PCA" + 10*"-")
print("Class 5 Accuracy: {:.2f}".format(acc5))
print("Class 6 Accuracy: {:.2f}".format(acc6))
print("Class 8 Accuracy: {:.2f}".format(acc8))
print("Total Accuracy: {:.2f}".format(acc))

# %% BONUS (PCA)
im5 = train5.reshape((-1, 28, 28))
im6 = train6.reshape((-1, 28, 28))
im8 = train8.reshape((-1, 28, 28))

inv_im5 = pca.inverse_transform(x5).reshape((-1, 28, 28))
inv_im6 = pca.inverse_transform(x6).reshape((-1, 28, 28))
inv_im8 = pca.inverse_transform(x8).reshape((-1, 28, 28))

fig, axes = plt.subplots(2, 3)
axes[0, 0].imshow(im5[0], cmap="Greys_r")
axes[0, 0].set_ylabel("True Images")
axes[1, 0].imshow(inv_im5[0], cmap="Greys_r")
axes[1, 0].set_ylabel("Inversed PCA Images")
axes[1, 0].set_xlabel("Class: 5")

axes[0, 1].imshow(im6[0], cmap="Greys_r")
axes[1, 1].imshow(inv_im6[0], cmap="Greys_r")
axes[1, 1].set_xlabel("Class: 6")

axes[0, 2].imshow(im8[0], cmap="Greys_r")
axes[1, 2].imshow(inv_im8[0], cmap="Greys_r")
axes[1, 2].set_xlabel("Class: 8")

# %%Part 1: Lower-Dimensional Reduction (LDA)
# The LDA class in scikit-learn fits a covariance matrix and compute
# eigenvectors for you. LDA assume that you know about the classes, so
# you have to use the concatenated training set and targets/classes
lda = LDA(n_components=2)
lda.fit(trainset, trn_targets)

# Lower-dimensional reduction of the training sets.
x5 = lda.transform(train5)
x6 = lda.transform(train6)
x8 = lda.transform(train8)

# Scatter plot of the dimensional-reduced data
fig, ax = plt.subplots(figsize=(12, 8))
fig.suptitle("LDA")
ax.set_axisbelow(True)
ax.grid()
ax.scatter(x5[:, 0], x5[:, 1], c='k', marker="x", label="Class 5")
ax.scatter(x6[:, 0], x6[:, 1], c='r', marker="x", label="Class 6")
ax.scatter(x8[:, 0], x8[:, 1], c='b', marker="x", label="Class 8")
ax.legend()

# Lower-dimensional reduction of the training sets.
tst = lda.transform(testset)

# %%Part 2: Classification (LDA)
# Estimate parameters for a bivariante Gaussian distribution.
mean5 = np.mean(x5, axis=0)
mean6 = np.mean(x6, axis=0)
mean8 = np.mean(x8, axis=0)

cov5 = np.cov(x5.T)
cov6 = np.cov(x6.T)
cov8 = np.cov(x8.T)

# Densities
l5 = norm(mean5, cov5)
l6 = norm(mean6, cov6)
l8 = norm(mean8, cov8)

# Compute a posterior probabilities
p5 = l5.pdf(tst)
p6 = l6.pdf(tst)
p8 = l8.pdf(tst)

# Compute predictions
pred = np.argmax(np.c_[p5, p6, p8], axis=1)
pred = classes[pred]

# Compute accuracy
acc5 = np.sum(pred[tst_targets == 5] == 5)/len(tst_target5) * 100
acc6 = np.sum(pred[tst_targets == 6] == 6)/len(tst_target6) * 100
acc8 = np.sum(pred[tst_targets == 8] == 8)/len(tst_target8) * 100
acc = np.sum(pred == tst_targets)/len(tst_targets) * 100
print("-"*10 + "LDA" + 10*"-")
print("Class 5 Accuracy: {:.2f}".format(acc5))
print("Class 6 Accuracy: {:.2f}".format(acc6))
print("Class 8 Accuracy: {:.2f}".format(acc8))
print("Total Accuracy: {:.2f}".format(acc))
