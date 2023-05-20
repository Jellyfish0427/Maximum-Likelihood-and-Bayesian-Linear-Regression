# Maximum-Likelihood-and-Bayesian-linear-regression
Apply the Maximum Likelihood (ML) and Bayesian linear regression methods to train a linear model in order to predict the calories burnt during exercise.

## Data
Contains two *.csv files, [exercise.csv] and [calories.csv].  
The exercise.csv have 15000 pieces of data in total (X)  
The calories.csv have 15000 pieces of data in total (Y)  
Merge them and Split them into 70:10:20 for training, validation, and testing, respectively.

## Implement
Employ the linear model to predict calories burnt with respect to exercise in the testing set.  
1. Use Maximum Likelihood Estimation to train the model. Then, use trained linear model to predict the burnt calories and compute the mean squared error for each data in testing_set.
2.  Use Bayesian Linear Regression to estimate w. Then, use estimated parameter to predict the burnt calories and compute the mean squared error for each data in validation_set.
3.  Plot the best fit lines for both models.
4.  Implement any regression model to get the best possible MSE. (built-in libaries)
