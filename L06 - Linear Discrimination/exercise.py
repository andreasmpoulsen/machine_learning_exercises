# %% Imports
from scipy.io import loadmat
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal as multi_norm

data = loadmat('../Datasets/mnist_all.mat')

# Empty lists for data
trainset = []
train_class = []
testset = []
test_class = []

# For each class
for i in range(10):
    trainset.append(data["train%d" % i])
    train_class.append(np.full(len(data["train%d" % i]), i))
    testset.append(data["test%d" % i])
    test_class.append(np.full(len(data["test%d" % i]), i))

trainset = np.concatenate(trainset)
train_class = np.concatenate(train_class)
testset = np.concatenate(testset)
test_class = np.concatenate(test_class)

# %% For 2 and 9 features
n_components = 9
# n_components = 9

# Initialize and fit LDA
lda = LDA(n_components=n_components)
lda.fit(trainset, train_class)

# Initialize and fit PCA
pca = PCA(n_components=n_components)
pca.fit(trainset)

# Transforming data
train_lda = lda.transform(trainset)
test_lda = lda.transform(testset)
train_pca = pca.transform(trainset)
test_pca = pca.transform(testset)

# Estimating parameters for a bivariante Gaussian distribution
# For LDA
_, lda_d = train_lda.shape

lda_means = np.zeros((10, lda_d))
lda_covs = np.zeros((10, lda_d, lda_d))

for i in range(10):
    indx = train_class == i
    lda_means[i] = np.mean(train_lda[indx], axis=0)
    lda_covs[i] = np.cov(train_lda[indx].T)

# For PCA
_, pca_d = train_pca.shape

pca_means = np.zeros((10, pca_d))
pca_covs = np.zeros((10, pca_d, pca_d))

for i in range(10):
    indx = train_class == i
    pca_means[i] = np.mean(train_pca[indx], axis=0)
    pca_covs[i] = np.cov(train_pca[indx].T)

# Computing multivariate gaussian distribution
# For LDA
probs_lda = []
for i in range(len(lda_covs)):
    probs_lda.append(multi_norm.pdf(test_lda, lda_means[i], lda_covs[i]))
probs_lda = np.c_[tuple(probs_lda)]
preds_lda = np.argmax(probs_lda, axis=1)

# For PCA
probs_pca = []
for i in range(len(pca_covs)):
    probs_pca.append(multi_norm.pdf(test_pca, pca_means[i], pca_covs[i]))
probs_pca = np.c_[tuple(probs_pca)]
preds_pca = np.argmax(probs_pca, axis=1)

# Compute accuracy
# For LDA
lda_acc = np.sum(preds_lda == test_class)/len(test_class) * 100

# For PCA
pca_acc = np.sum(preds_pca == test_class)/len(test_class) * 100

print("n_components: 2")
print("PCA Accuracy: {:.2f}".format(pca_acc))
print("LDA Accuracy: {:.2f}".format(lda_acc))
