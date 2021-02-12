# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 21:27:23 2021

@author: Neel
"""
print("Importing necessary libraries like pandas,numpy and matplotlib(for plotting)...")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print("import done")

datapath = input('feed the path of the data you want to use decision tree on')
data = pd.read_csv(datapath)
print("\nthis is minimal view of the dataset you have selected")
print(data.head())

print("\n importing the sklearn library for the built in decision tree classifier ")
from sklearn.tree import DecisionTreeRegressor
print("\n import done")

print("\n spilting the data into two features and target variable assuming last column to be the target variable")
features = data.iloc[:,:-1]
print("\n the features of the dataset are: ")
print(features.head())

target = data.iloc[:,-1]
print("the target variable of the dataset is: ")
print(target.head())

print("\n histogram visualization for different feature relations of the dataset")
data.hist(bins=50,figsize=(20,15))
plt.show()

print("\n importing model selection for splitting train and test set")
from sklearn.model_selection import train_test_split
print("\n import done")
test_ratio = float(input('\n enter the ratio you want to split the data with'))
train_set,test_set = train_test_split(data,test_size=test_ratio,random_state=42)
print("\n this is the training set")
print(train_set)

train_features = train_set.iloc[:,:-1]
test_features  = test_set.iloc[:,:-1]
train_target = train_set.iloc[:,-1]
test_target = test_set.iloc[:,-1]

print("\nUsing decision tree regressor on the train set")
regressor = DecisionTreeRegressor(random_state = 42)
regressor.fit(train_features, train_target)

test_prediction = regressor.predict(test_features)

from sklearn.metrics import mean_squared_error
lin_mse = mean_squared_error(test_target,test_prediction)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)
print('\nThis is the mean squared error of of the decision tree on the test set')
