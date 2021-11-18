# Exercise - Algorithm-independent Machine Learning and Reinforcement Learning

In this exercise we examine the use of decision trees for both classification and regression. First import [the Boston Housing dataset](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html) and split the data into a training set and a test set.

For classification:

Artificially create two classes based on the price. Use mean price as the threshold: class 1 if the price is larger than mean; otherwise, class 0. Inspiration can be found in previous examples of Algorithm Illustration by Code. Split the data into training (80%) and test (20%). Afterwards train a decision tree classifier on the training data and see how it performs on both the training data and the test data. Try to experiment with the depth of the tree and see what effect this has on the classification accuracy. Note all 13 dimensional features, as originally provided by the dataset, are used as input, and the labels/targets are 0 (low price) and 1 (high price).

For regression:

Train a decision tree regressor on the data with prices as the targets. As the classifier case, try to examine how the depth of your trees affect your results. Again, all 13 dimensional features are used as input, and the prices as the labels/targets.
