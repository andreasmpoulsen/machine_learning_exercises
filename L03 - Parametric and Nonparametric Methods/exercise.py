# %% Loading data

from numpy.core.fromnumeric import argmax
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as multi_norm

data = loadmat("dataset1_G_noisy.mat")

trn_x = data['trn_x']
trn_y = data['trn_y']
trn_x_class = data['trn_x_class']
trn_y_class = data['trn_y_class']

tst_xy = data["tst_xy"]
tst_xy_class = data["tst_xy_class"]
tst_xy_126 = data["tst_xy_126"]
tst_xy_126_class = data["tst_xy_126_class"]

# %% 1. classify instances in tst_xy, and use the corresponding label
#  file tst_xy_class to calculate the accuracy

# Number of training samples:
N = len(trn_x) + len(trn_y)

# Estimated prior:
prior_x = len(trn_x)/N
prior_y = 1-prior_x

# For using a multivariate gaussian, we estimate mean and covariance
# for X and Y:
mean_x = np.mean(trn_x, axis=0)
mean_y = np.mean(trn_y, axis=0)
cov_x = np.cov(trn_x.T)
cov_y = np.cov(trn_y.T)

# Multivariate normals of X and Y:
norm_x = multi_norm(mean=mean_x, cov=cov_x)
norm_y = multi_norm(mean=mean_y, cov=cov_y)

# Posterior probability on the test
pos_x = prior_x * norm_x.pdf(tst_xy)
pos_y = prior_y * norm_y.pdf(tst_xy)

# Maximum a posteriori prediction, if pos_x > pos_y, predict class 1
# otherwise predict class 2
pred = np.argmax(np.c_[pos_x, pos_y], axis=1) + 1

# Compute accuracy
true = tst_xy_class.squeeze()
acc = np.sum(pred == true)/len(true) * 100
print("1.: Accuracy: {:.2f}".format(acc))
# %% 2. classify instances in tst_xy_126 by assuming a
# uniform prior over the space of hypotheses, and use the
# corresponding label file tst_xy_126_class to calculate the accuracy

# Uniform prior:
prior_x = 0.5
prior_y = 0.5

# Posterior probability on the test
pos_x = prior_x * norm_x.pdf(tst_xy_126)
pos_y = prior_y * norm_y.pdf(tst_xy_126)

# Maximum a posteriori prediction, if pos_x > pos_y, predict class 1
# otherwise predict class 2
pred = np.argmax(np.c_[pos_x, pos_y], axis=1) + 1

# Compute accuracy
true = tst_xy_126_class.squeeze()
acc = np.sum(pred == true)/len(true) * 100
print("2.: Accuracy: {:.2f}".format(acc))

# %% 3. classify instances in tst_xy_126 by assuming a
# prior probability of 0.9 for Class x and 0.1 for Class y,
# and use the corresponding label file tst_xy_126_class
# to calculate the accuracy; compare the results with those of (2).

# Prior probability:
prior_x = 0.9
prior_y = 0.1

# Posterior probability on the test
pos_x = prior_x * norm_x.pdf(tst_xy_126)
pos_y = prior_y * norm_y.pdf(tst_xy_126)

# Maximum a posteriori prediction, if pos_x > pos_y, predict class 1
# otherwise predict class 2
pred = np.argmax(np.c_[pos_x, pos_y], axis=1) + 1

# Compute accuracy
true = tst_xy_126_class.squeeze()
acc = np.sum(pred == true)/len(true) * 100
print("3.: Accuracy: {:.2f}".format(acc))
