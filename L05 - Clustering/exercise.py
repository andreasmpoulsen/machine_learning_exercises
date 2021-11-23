# %% Loading data

from scipy.io import loadmat
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal as multi_norm
import matplotlib.pyplot as plt

data = loadmat('Datasets/2D568class.mat')

trn5 = data['trn5_2dim']/255
trn6 = data['trn6_2dim']/255
trn8 = data['trn8_2dim']/255

# %% 1. mix the 2-dimensional data (training data only)
# by removing the labels and then use one Gaussian mixture model
# to model them

# Mixing the data

trn_data = np.concatenate([trn5, trn6, trn8])
np.random.shuffle(trn_data)

gm = GaussianMixture(n_components=3).fit(trn_data)

# %% 2. compare the Gaussian mixture model with the Gaussian models
# trained in the previous assignment, in terms of mean and variance
# values as well as through visualisation.

# Estimating mean and variance for multivariate gaussian
mean5 = np.mean(trn5, axis=0)
mean6 = np.mean(trn6, axis=0)
mean8 = np.mean(trn8, axis=0)

cov5 = np.cov(trn5.T)
cov6 = np.cov(trn6.T)
cov8 = np.cov(trn8.T)

# Computing multivariate gaussian distribution
norm5 = multi_norm(mean5, cov5)
norm6 = multi_norm(mean6, cov6)
norm8 = multi_norm(mean8, cov8)

# Create points to do a contour a plot
x = np.linspace(-8., 8., 101)
y = np.linspace(-7.2, 7.2, 101)
X, Y = np.meshgrid(x, y)
r = np.c_[X.ravel(), Y.ravel()]

# Get means and variances from GMM to create new
# multivariate gaussian distributions
norm_mean5, norm_mean6, norm_mean8 = gm.means_
norm_cov5, norm_cov6, norm_cov8 = gm.covariances_

# Creating multivariate gaussian distributions from GMM means and variances
gmm_norm5 = multi_norm(norm_mean5, norm_cov5)
gmm_norm6 = multi_norm(norm_mean6, norm_cov6)
gmm_norm8 = multi_norm(norm_mean8, norm_cov8)

## THE FOLLOWING WAS TAKEN FROM THE EXAMPLE ##

# Compute the probabilities to greate contours
Z = np.exp(gm.score_samples(r).reshape(X.shape))

Z1 = gmm_norm5.pdf(r).reshape(X.shape)
Z2 = gmm_norm6.pdf(r).reshape(X.shape)
Z3 = gmm_norm8.pdf(r).reshape(X.shape)

Z5 = norm5.pdf(r).reshape(X.shape)
Z6 = norm6.pdf(r).reshape(X.shape)
Z8 = norm8.pdf(r).reshape(X.shape)

# Plot contours for the GMM, seperated GMM and individual estimated densities
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
cntr = ax1.contourf(X, Y, Z, levels=np.linspace(0., 0.045, 11))
ax1.contour(X, Y, Z, colors="k", levels=np.linspace(0., 0.045, 11))
fig.colorbar(cntr, ax=ax1)
ax1.set_title("Gaussian Mixture Model (GMM)")

ax2.set_axisbelow(True)
ax2.grid()
ax2.contour(X, Y, Z1, colors="k")
ax2.contour(X, Y, Z2, colors="r")
ax2.contour(X, Y, Z3, colors="b")
ax2.set_title("Seperated GMM")

ax3.set_axisbelow(True)
ax3.grid()
cntr1 = ax3.contour(X, Y, Z5, colors="k")
cntr2 = ax3.contour(X, Y, Z6, colors="r")
cntr3 = ax3.contour(X, Y, Z8, colors="b")
h1, _ = cntr1.legend_elements()
h2, _ = cntr2.legend_elements()
h3, _ = cntr3.legend_elements()
ax3.legend([h1[0], h2[0], h3[0]], ["Class 5", "Class 6", "Class 8"])
ax3.set_title("Esitmated Densities for each class")

plt.show()
