import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import random
from numpy.linalg import inv
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

np.random.seed(42)

train_ratio = 0.7
valid_ratio = 0.1
test_ratio = 0.2

def split_data(feature, label):
    total_size = len(feature)
    train_num = int((total_size) * train_ratio) #10500
    valid_num = int((total_size) * valid_ratio) #1500
    test_num = int((total_size) * test_ratio) #3000

    feature_label_array = np.column_stack((feature, label))
    np.random.shuffle(feature_label_array)  #shuffle

    train_array = feature_label_array[0 : train_num]
    valid_array = feature_label_array[train_num : train_num+valid_num]
    test_array = feature_label_array[train_num+valid_num : total_size]

    train_feature = train_array[:, :-1].astype(float)  
    train_label = train_array[:, -1].astype(float)
    valid_feature = valid_array[:, :-1].astype(float)
    valid_label = valid_array[:, -1].astype(float)
    test_feature = test_array[:, :-1].astype(float)
    test_label = test_array[:, -1].astype(float)

    return train_feature, train_label, valid_feature, valid_label, test_feature, test_label 

def add_bias(data):
    return np.concatenate((np.ones((data.shape[0], 1)), data), axis=1) 
    
def MSE(y_pred, y_label):
    return np.mean((y_label-y_pred)**2)

def MLR(train_feature, train_label, test_feature):
    train_feature = add_bias(train_feature)
    test_feature = add_bias(test_feature)
    weights = np.linalg.inv(train_feature.T @ train_feature) @ train_feature.T @ train_label
    pred = test_feature @ weights

    return pred, weights

def BLR(train_feature, train_label, test_feature, alpha, beta):
    train_feature = add_bias(train_feature)
    test_feature = add_bias(test_feature)
    S_inv = alpha * np.eye(train_feature.shape[1]) + beta * train_feature.T @ train_feature
    S = inv(S_inv)
    weights = beta * S @ train_feature.T @ train_label
    pred = test_feature @ weights

    return pred, weights

def plot_obs(x, y):
    plt.scatter(x, y, label='Observations', s=3, c='b')
    plt.plot()
    plt.xlabel('Duration(min)')
    plt.ylabel('Calories')     

def MLR_fit_line(x, y, weights):
    # MLR fit line 
    y_MLR = weights[5] * x + weights[7]
    plt.plot(x, y_MLR, label="OLS Fit", linestyle='--', color="black")

def BLR_fit_line(train_feature, train_label, valid_feature):
    num_samples = 100
    a_start, a_stop = 1e+2, 1e-2
    a_array = np.logspace(np.log10(a_start), np.log10(a_stop), num=num_samples)

    b_start, b_stop = 1e-2, 1e+2
    b_array = np.logspace(np.log10(b_start), np.log10(b_stop), num=num_samples)

    for i in range(len(a_array)):
        alpha = a_array[i]
        beta = b_array[i]
        pred, weights = BLR(train_feature, train_label, valid_feature, alpha, beta)

        y_BLR = weights[5] * train_feature[:, 4] + weights[7]
        if i == 0:
            plt.plot(train_feature[:, 4], y_BLR, label="Bayesian Posterior fits", color="red")
        else:
            plt.plot(train_feature[:, 4], y_BLR, color="red")
        
def model_train(model,x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    model_pred = model.predict(x_test)
    model_mse = MSE(model_pred, y_test)
    print('MSE of {} = {}'.format(model,model_mse))

def main():
    # read data
    feature_df = pd.read_csv("/Users/jellyfish/Desktop/ML/HW3/exercise.csv", sep=",", header=None)
    label_df = pd.read_csv("/Users/jellyfish/Desktop/ML/HW3/calories.csv", sep=",", header=None)

    # preprocess
    # 1. male: 0, female: 1
    feature_df.iloc[1:,1] = feature_df.iloc[1:,1].replace({'male': 0, 'female': 1}) # male:0, female:1

    # 2. delete ID
    feature_df = feature_df.drop(0,axis=1)
    label_df = label_df.drop(0,axis=1)

    # 3. delete label
    feature_df = feature_df.drop(0,axis=0)
    label_df = label_df.drop(0,axis=0)

    # dataframe to array
    feature_df_values = feature_df.values
    label_df_values = label_df.values

    # split data
    train_feature, train_label, valid_feature, valid_label, test_feature, test_label = split_data(feature_df_values, label_df_values)

    # MLR
    MLR_pred, MLR_weights = MLR(train_feature, train_label, test_feature)
    MLR_MSE = MSE(MLR_pred, test_label)
    print('MSE of MLR = {}'.format(MLR_MSE))
    
    # BLR
    BLR_pred, BLR_weights = BLR(train_feature, train_label, valid_feature, alpha=1, beta = 1/(0.1 ** 2) )
    BLR_MSE = MSE(BLR_pred, valid_label)
    print('MSE of BLR = {}'.format(BLR_MSE))

    # Implement regression models
    model_train(RandomForestRegressor(),train_feature, train_label, test_feature, test_label)
    model_train(LinearRegression(),train_feature, train_label, test_feature, test_label)
    model_train(SVR(),train_feature, train_label, test_feature, test_label)
    model_train(KNeighborsRegressor(),train_feature, train_label, test_feature, test_label)

    # Ｐlot MLR and BLR fit line
    plot_obs(train_feature[:, 4], train_label)  
    BLR_fit_line(train_feature, train_label, valid_feature)
    MLR_fit_line(train_feature[:, 4], train_label, MLR_weights)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()