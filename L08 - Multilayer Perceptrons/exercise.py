# %% Loading data
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

n_components = 30

data = loadmat('../Datasets/mnist_all.mat')

# Extracting training sets from data dict
train5 = data['train5']/255
train6 = data['train6']/255
train8 = data['train8']/255

# Create classes for training sets
train5_class = 5*np.ones(len(train5))
train6_class = 6*np.ones(len(train6))
train8_class = 8*np.ones(len(train8))

# Extracting test sets from data dict
test5 = data['test5']/255
test6 = data['test6']/255
test8 = data['test8']/255

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
pca = PCA(n_components=n_components)
pca.fit(train_data)

# Transform the data from 784 to 2 features
new_train_data = pca.transform(train_data)
new_test_data = pca.transform(test_data)

# %% Develop an MLP for the MNIST database by using the dimension-reduced data from your work on Exercises 2 and 3

clf = MLPClassifier(random_state=1, max_iter=1000, solver='adam').fit(
    new_train_data, train_data_classes)
print("Acc {}-dim: {:.2f}".format(n_components,
      clf.score(new_test_data, test_data_classes)))
