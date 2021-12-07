# %% Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Artificially create two classes based on the price.
# Use mean price as the threshold: class 1 if the price is
# larger than mean; otherwise, class 0.

price_threshold = np.mean(target)

target_binary = []

for i in range(target.size):
    if target[i] > price_threshold:
        target_binary.append(1)
    else:
        target_binary.append(0)

# Split the data into training (80%) and test (20%)

train_data, test_data, train_target, test_target = train_test_split(
    data, target_binary, test_size=0.2, random_state=42)


# %% Train a decision tree classifier on the training data and
# see how it performs on both the training data and the test data
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3)

# Fit model
clf.fit(train_data, train_target)

# Predict response for test set
pred = clf.predict(test_data)

# Get accuracy of model based on test data
acc = clf.score(test_data, test_target)
print(acc)

# %% Train a decision tree regressor on the data with prices as the targets.
# As the classifier case, try to examine how the depth of your
# trees affect your results. Again, all 13 dimensional features
# are used as input, and the prices as the labels/targets.
regressor = DecisionTreeRegressor(max_depth=3)

# Fit model
regressor.fit(train_data, train_target)

# Predict response for test set
pred = regressor.predict(test_data)

# Get accuracy of model based on test data
acc = regressor.score(test_data, test_target)
print(acc)
