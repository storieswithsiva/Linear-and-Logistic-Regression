# Linear-and-Logistic-Regression
Linear and Logistic Regression

# Linear-Regression-and-Logistic-Regression
This is my implementation of Linear Regression and Logistic Regression using python and numpy.

In this work, I used two LIBSVM datasets which are pre-processed data originally from UCI data repository.

1. Linear regression - Housing dataset (housing scale dataset). Predict housing values in suburbs of Boston. https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/housing_scale.
2. Logistic regression - Adult dataset (I only use a3a training dataset). Predict whether income exceeds $50K/yr based on census data. https: //www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a3a

## Problem 1
**Linear regression**. I randomly split the dataset into two groups: training (around 80%) and testing (around 20%). Then I learn the linear regression model on the training data, using the analytic solution. After I compute the prediction error on the test data. I repeat this process 10 times and report all individual prediction errors of 10 trials and the average of them.

## Problem 2
**Linear regression**. I do the same work as in the problem #1 but now using a gradient descent. (10 randomly generated datasets in #1 should be maintained; we will use the datasets generated in #1.) Here I am not using (exact or backtracking) line searches. I try several selections for the fixed step size. 

## Problem 3
**Logistic regression**. As in the problem #1, I randomly split the adult dataset into two groups (80% for training and 20% testing). Then I learn logistic regression on the training data. Here I compare the performances of gradient descent methods i) with fixed-sized step sizes and ii) with the backtracking line search. I tried to find the best step size for i) and the best hyperparameters α and β for ii) (in terms of the final objective function values).
