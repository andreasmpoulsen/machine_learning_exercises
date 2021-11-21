# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 09:22:05 2020

@author: Morten Østergaard Nielsen
"""

# %%Load dataset
import numpy as np
from scipy.io import loadmat
from scipy.stats import multivariate_normal as norm
import matplotlib.pyplot as plt

file = "dataset1_G_noisy.mat"
data = loadmat(file)

# Trainsets
trn_x = data["trn_x"]
trn_y = data["trn_y"]
trn_x_class = data["trn_x_class"]
trn_y_class = data["trn_y_class"]

# Testsets
tst_xy = data["tst_xy"]
tst_xy_class = data["tst_xy_class"]
tst_xy_126 = data["tst_xy_126"]
tst_xy_126_class = data["tst_xy_126_class"]

# Plot training data
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_axisbelow(True)
ax.grid()
ax.scatter(trn_x[:, 0], trn_x[:, 1], c='b', marker=".", label="Class 1")
ax.scatter(trn_y[:, 0], trn_y[:, 1], c='r', marker=".", label="Class 2")
ax.legend()

"""
From the plot it looks like the data points from both classes are 
Gaussian/Normal distributed. So in order to estimate a probability density
for the training set, an obvious choice will be an parametric method.
"""

# %% Part a)
# Total number of training samples
N = len(trn_x_class) + len(trn_y_class)

# Estimated Prior
# Alternative to the uniform prior. Estimate priors from the training set.
# prior_x = len(trn_x_class)/N
# prior_y = len(trn_y_class)/N

# Uniform Prior
prior_x = 0.5
prior_y = 0.5

# Estimate mean value and covariance for X and Y
mean_x = np.mean(trn_x, axis=0)
mean_y = np.mean(trn_y, axis=0)
cov_x = np.cov(trn_x.T)
cov_y = np.cov(trn_y.T)

# Create multivariate gaussian distributions for X and Y
l_x = norm(mean=mean_x, cov=cov_x)
l_y = norm(mean=mean_y, cov=cov_y)

# Add pdf contours to the plot
x = np.linspace(-7.5, 3.5, 101)
y = np.linspace(-2.5, 1.5, 101)
X, Y = np.meshgrid(x, y)
z = np.c_[X.ravel(), Y.ravel()]
P1 = l_x.pdf(z).reshape(X.shape)
P2 = l_y.pdf(z).reshape(X.shape)
ax.contour(X, Y, P1, 6, colors='k')
ax.contour(X, Y, P2, 6, colors='k')

# Posteriori probability on the test set
p1 = prior_x * l_x.pdf(tst_xy)
p2 = prior_y * l_y.pdf(tst_xy)

# Maximum a posteriori prediction
# If p1 > p2 we predict class 1 otherwise it is class 2.
pred = np.argmax(np.c_[p1, p2], axis=1) + 1

# Compute accuracy. Count total of correct prediction.
true = tst_xy_class.squeeze()
acc = np.sum(pred == true)/len(true) * 100
print("a) Accuracy: {:.2f}".format(acc))

# %% Part b)
# Uniform Prior
prior_x = 0.5
prior_y = 0.5

# Posteriori probability on the test set
p1 = prior_x * l_x.pdf(tst_xy_126)
p2 = prior_y * l_y.pdf(tst_xy_126)

# Maximum a posteriori prediction
pred = np.argmax(np.c_[p1, p2], axis=1) + 1

# Compute accuracy
true = tst_xy_126_class.squeeze()
acc = np.sum(pred == true)/len(true) * 100
print("b) Accuracy: {:.2f}".format(acc))

# %% Part c)
# Skrewed prior
prior_x = 0.9
prior_y = 0.1

# Posteriori probability on the test set
p1 = prior_x * l_x.pdf(tst_xy_126)
p2 = prior_y * l_y.pdf(tst_xy_126)

# Maximum a posteriori prediction
pred = np.argmax(np.c_[p1, p2], axis=1) + 1

# Compute accuracy
true = tst_xy_126_class.squeeze()
acc = np.sum(pred == true)/len(true) * 100
print("c) Accuracy: {:.2f}".format(acc))
