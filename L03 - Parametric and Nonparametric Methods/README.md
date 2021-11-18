# Exercise - Parametric and Nonparametric Methods

With dataset1_noisy (generated from handwritten digits database) available in this folder, do the following exercise (using any program language you like, e.g., Python, Matlab): You are given, as the train data, trn_x and trn_y along with their class labels trn_x_class and trn_y_class.

The task is to classify the following TEST data.

1. classify instances in tst_xy, and use the corresponding label file tst_xy_class to calculate the accuracy;
2. classify instances in tst_xy_126 by assuming a uniform prior over the space of hypotheses, and use the corresponding label file tst_xy_126_class to calculate the accuracy;
3. classify instances in tst_xy_126 by assuming a prior probability of 0.9 for Class x and 0.1 for Class y, and use the corresponding label file tst_xy_126_class to calculate the accuracy; compare the results with those of (2).

To plot Gaussian, you may use [PLOT_GAUSSIAN_ELLIPSOID](https://www.mathworks.com/matlabcentral/fileexchange/16543-plot_gaussian_ellipsoid).
