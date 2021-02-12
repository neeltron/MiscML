#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[2]:


dataset = input("Enter the dataset: ")
df = pd.read_csv(dataset)


# In[3]:


df =df.drop("Address", axis=1)


# In[4]:


df.head()


# In[5]:


def checkNull():
    
    print("Checking for null values...\n")
    count = 0
    
    for i in df.columns:
        if(i == np.where(pd.isnull(i))):
            count += 1
    if (count>0):
        print(str(count) + "columns contain null values")
    else:
        print("No column contains null values")
    
    return 


# In[6]:


def splitFrame():
    
    print("Spilting the data into two features and target variable assuming last column to be the target variable...")
    
    X = df.iloc[:,: -1]
    y = df.iloc[:,-1]
    
    return (X,y)


# In[7]:


def tr_ts_split(X,y):
    
    print("\nPerforming train-test split... (Default train size = 0.7)")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    
    return (X_train, X_test, y_train, y_test)


# In[8]:


def makePred(X,lm):
    
    vals = []
    
    for i in range(X.shape[1]):
        a = float(input("Enter value for column " + str(i) + ": "))
        vals.append(a)
        
    pred = lm.predict([vals])
    
    print("---------------------------------------------")
    print("\nPredicted value: ", pred)


# In[9]:


def linarReg():
    
    checkNull()
    print("\nPerforming Linear Regression on the given dataset.")
    print("----------------------------------------------")
    
    X,y = splitFrame()
    X_train, X_test, y_train, y_test = tr_ts_split(X,y)
    
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    predictions = lm.predict(X_test)
    
    print("\nTraining the model..")
    print("\nTestting the model..")
    
    print("\nTask completed.")
    print("----------------------------------------------")
    print("Displaying results:\n")
    
    accuracy = lm.score(X_test,y_test)
    print("Accuracy: ", accuracy*100,'%\n')
    
    print('MAE:', metrics.mean_absolute_error(y_test, predictions))
    print('MSE:', metrics.mean_squared_error(y_test, predictions))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
    
    
    coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
    display(coeff_df)
    
    ch = input("Make a prediciton (Y/N)?: ")
    
    if ch =="Y" or ch == "y":
        makePred(X,lm)
    
    else:
        print("Okay!")


# In[10]:


linarReg()


# In[ ]:




