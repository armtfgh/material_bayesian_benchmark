from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor, BayesianRidge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, plot_confusion_matrix, accuracy_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from xgboost import XGBRegressor
from numpy import absolute,std, mean
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
# import tensorflow as tf
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
import math




df = pd.read_csv('data.csv')
data  = df.to_numpy()
X = data[:,:4]
Y = data[:,4]
scaler = StandardScaler()


def baseline_model(x,y):
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

    train_scaled = scaler.fit_transform(X_train, y_train)
    test_scaled = scaler.transform(X_test)
    regressor=MLPRegressor(max_iter=500,solver='lbfgs',learning_rate='constant',activation='relu',hidden_layer_sizes=(61,81,62))
    regressor.fit(train_scaled,y_train)
    y_pred = regressor.predict(test_scaled)
    mae = mean_absolute_error(y_test,y_pred)
    rmse = mean_squared_error(y_test,y_pred)**0.5
    r2 = r2_score(y_test,y_pred)
    return [mae,rmse,r2,y_pred,y_test]

def baseline_model2(x,y):
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

    train_scaled = scaler.fit_transform(X_train, y_train)
    test_scaled = scaler.transform(X_test)
    regressor=XGBRegressor()
    regressor.fit(train_scaled,y_train)
    y_pred = regressor.predict(test_scaled)
    mae = mean_absolute_error(y_test,y_pred)
    rmse = mean_squared_error(y_test,y_pred)**0.5
    r2 = r2_score(y_test,y_pred)
    return [mae,rmse,r2,y_pred,y_test]


s=baseline_model(X, Y)
print(s[0])



    
    