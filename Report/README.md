## 1.	Maximum Likelihood Estimation
![截圖 2023-06-02 上午2 29 09](https://github.com/Jellyfish0427/Maximum-Likelihood-and-Bayesian-linear-regression/assets/128220508/ef489fa7-9005-4165-9185-1294fcc2fcc9).  
To calculate the weights of the model, we organize the features into the matrix Φ, and the corresponding labels of the training data into the vector t.
```js
def MLR(train_feature, train_label, test_feature):
    train_feature = add_bias(train_feature)
    test_feature = add_bias(test_feature)
    weights = np.linalg.inv(train_feature.T @ train_feature) @ train_feature.T @ train_label
    pred = test_feature @ weights

    return pred, weights
```
#### MSE = 127.9584

## 2. Bayesian Linear Regression
We utilize the prior distribution, and the observed data to compute the weights of the model. 
The prior probability distribution is represented as 𝑝(𝑤) =𝑁(𝑤|𝑚0, 𝑆0),where m_0 is the mean and S_0 is the covariance matrix.

We incorporate the prior information by setting   ![截圖 2023-06-02 上午2 50 10](https://github.com/Jellyfish0427/Maximum-Likelihood-and-Bayesian-linear-regression/assets/128220508/7f53d410-f8b7-4b92-a651-295ab7b4a582)
, where α is the parameter of the prior distribution.  
By simplifying the equations, we can obtain the weights.    

![截圖 2023-06-02 上午2 47 59](https://github.com/Jellyfish0427/Maximum-Likelihood-and-Bayesian-linear-regression/assets/128220508/bb1acc80-cad8-4804-b56b-5ac8fa844fdd).     
If consider 𝛼 → 0, the result of BLR is same as MLR.   

```js
def BLR(train_feature, train_label, test_feature, alpha, beta):
    train_feature = add_bias(train_feature)
    test_feature = add_bias(test_feature)
    S_inv = alpha * np.eye(train_feature.shape[1]) + beta * train_feature.T @ train_feature
    S = inv(S_inv)
    weights = beta * S @ train_feature.T @ train_label
    pred = test_feature @ weights

    return pred, weights
```
#### MSE = 123.3814

## 3. Discuss the difference between Maximum Likelihood and Bayesian Linear Regression
![image](https://github.com/Jellyfish0427/Maximum-Likelihood-and-Bayesian-linear-regression/assets/128220508/c2588a83-2c6f-4d10-b517-08d03372613d)   
It can be found that the formulas for calculating the weights of the two are very similar, the difference is BLR has calculated the prior distributions.   
Regularization is naturally integrated into BLR through prior distribution. By specifying appropriate priors, it is possible to introduce regularization and control the complexity of the model and avoid overfitting.

## 4. Implement regression models
#### (1) Random Forest
MSE = 9.7459

#### (2) Linear Regression
MSE = 127.9584

#### (3) Support Vector Regression
MSE = 264.6485

#### (4) K-Nearest Neighbors Regression
MSE = 56.4326



