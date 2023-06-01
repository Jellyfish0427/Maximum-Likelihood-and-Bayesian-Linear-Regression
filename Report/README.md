## 1.	Maximum Likelihood Estimation
![截圖 2023-06-02 上午2 29 09](https://github.com/Jellyfish0427/Maximum-Likelihood-and-Bayesian-linear-regression/assets/128220508/ef489fa7-9005-4165-9185-1294fcc2fcc9).  
To calculate the weights of the model, we organize the features into the matrix Φ, and the corresponding labels of the training data into the vector t.
```js
def MLR(train_feature, train_label, test_feature):
    train_feature = add_bias(train_feature)
    test_feature = add_bias(test_feature)
    weights = np.linalg.inv(train_feature.T @ train_feature) @ train_feature.T @ train_label


## 2. Bayesian Linear Regression
