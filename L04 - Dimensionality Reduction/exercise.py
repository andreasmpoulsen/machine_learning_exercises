# %% Loading data
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal as multi_norm

data = loadmat('../Datasets/mnist_all.mat')

# %% 1. from the 10-class database, choose three classes (5, 6 and 8)
# and then reduce dimension to 2

# Extracting training sets from data dict
train5 = data['train5']
train6 = data['train6']
train8 = data['train8']

# Create classes for training sets
train5_class = 5*np.ones(len(train5))
train6_class = 6*np.ones(len(train6))
train8_class = 8*np.ones(len(train8))

# Extracting test sets from data dict
test5 = data['test5']
test6 = data['test6']
test8 = data['test8']

# Create classes for test sets
test5_class = 5*np.ones(len(test5))
test6_class = 6*np.ones(len(test6))
test8_class = 8*np.ones(len(test8))

# Mixing data and classes
train_data = np.append(train5, np.append(train6, train8, axis=0), axis=0)
train_data_classes = np.append(train5_class, np.append(
    train6_class, train8_class, axis=0), axis=0)

test_data = np.append(test5, np.append(test6, test8, axis=0), axis=0)
test_data_classes = np.append(test5_class, np.append(
    test6_class, test8_class, axis=0), axis=0)

# Make and fit PCA model
pca = PCA(n_components=2)
pca.fit(train_data)

# Transform the data from 784 to 2 features
new_train5 = pca.transform(train5)
new_train6 = pca.transform(train6)
new_train8 = pca.transform(train8)
new_test_data = pca.transform(test_data)

# %% 1. perform 3-class classification based on the generated
# 2-dimensional data

# Parameters for a bivariante Gaussian distribution
mean5 = np.mean(new_train5, axis=0)
mean6 = np.mean(new_train6, axis=0)
mean8 = np.mean(new_train8, axis=0)

cov5 = np.cov(new_train5.T)
cov6 = np.cov(new_train6.T)
cov8 = np.cov(new_train8.T)

# Computing multivariate gaussian distribution
norm5 = multi_norm(mean5, cov5)
norm6 = multi_norm(mean6, cov6)
norm8 = multi_norm(mean8, cov8)

# Computing a posterior probabilities
p5 = norm5.pdf(new_test_data)
p6 = norm6.pdf(new_test_data)
p8 = norm8.pdf(new_test_data)

# Predicting classes
pred = np.argmax(np.c_[p5, p6, p8], axis=1)
pred = np.array([5, 6, 8])[pred]

# Compute accuracy
acc5 = np.sum(pred[test_data_classes == 5] == 5)/len(test5_class) * 100
acc6 = np.sum(pred[test_data_classes == 6] == 6)/len(test6_class) * 100
acc8 = np.sum(pred[test_data_classes == 8] == 8)/len(test8_class) * 100
acc_total = np.sum(pred == test_data_classes)/len(test_data_classes) * 100
print("-"*10 + "PCA" + 10*"-")
print("Class 5 Accuracy: {:.2f}".format(acc5))
print("Class 6 Accuracy: {:.2f}".format(acc6))
print("Class 8 Accuracy: {:.2f}".format(acc8))
print("Total Accuracy: {:.2f}".format(acc_total))
