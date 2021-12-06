# %% Imports
from scipy.io import loadmat
import numpy as np
from sklearn import svm

data = loadmat("../Datasets/mnist_all.mat")

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

trainset = np.concatenate(trainset)/255
train_class = np.concatenate(train_class)
testset = np.concatenate(testset)/255
test_class = np.concatenate(test_class)

# %% Initializing and fitting SVM
clf = svm.SVC(max_iter=1000)
clf.fit(trainset, train_class)

# Predicting on test set
pred = clf.predict(testset)

# Calculating accuracy
acc = np.sum(pred == test_class)/len(test_class) * 100

print("Accuracy: {:.2f}".format(acc))

# %%
