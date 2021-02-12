#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix

import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("Classified Data",index_col=0)
df.head()


# In[3]:


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


# In[4]:


def splitFrame():
    
    print("Spilting the data into two features and target variable assuming last column to be the target variable...")
    
    X = df.iloc[:,: -1]
    y = df.iloc[:,-1]
    
    return (X, y)


# In[5]:


def tr_ts_split(scaled_X,y):
    
    print("\nPerforming train-test split... (Default train size = 0.7)")
    
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.3, random_state=101)
    
    return (X_train, X_test, y_train, y_test)


# In[6]:


def printRes(k, y_test, pred):
    
    print("Displaying Results:")
    print('\nWITH K = ',k)
    print('\nConfusion Matrix')
    print('\n')
    print(confusion_matrix(y_test,pred))
    print('\n\nClassificaation Report')
    print('\n')
    print(classification_report(y_test,pred))


# In[7]:


def kPlot(X_train, X_test, y_test, y_train):
    
    plt.figure(figsize=(10,6))
    error_rate = []

    for i in range(1,51):
    
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train,y_train)
        pred_i = knn.predict(X_test)
        error_rate.append(np.mean(pred_i != y_test))

    plt.plot(range(1,51),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
    
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.show()


# In[8]:


def setK():
    k = 0
    res = 1
    ch = input("Choose another K value (Y/N)?: ")
    if (ch == "Y" or ch == "y"):
        k = int(input("Enter K: "))
    else:
        res = 0
        print("okay!")
    
    return (res,k)    


# In[14]:


def makePred(X, knn):

    ch = input("Make Prediction (Y/N)?: ")
    print("----------------------------------------------")
    if (ch=="Y" or ch =='y'):
        vals = []
        for i in range(X.shape[1]):
            val = float(input("Enter value for column " + str(i) + ": "))
            vals.append(val)
        print("----------------------------------------------")
        print("Predicted Class: ", knn.predict([vals]))
    
    else:
        print("okay!")


# In[10]:


def KNN():
    
    k = 15
    
    checkNull()
    
    print("\nStarting KNN..")
    print("----------------------------------------------")
    
    X, y = splitFrame()
    
    print("\nStandardizing variables..")
    scaler = StandardScaler()
    scaler.fit(X)
    scaled_X = scaler.transform(X)
    
    X_train, X_test, y_train, y_test = tr_ts_split(scaled_X, y)
    
    print("\n(Default K value is 15)")
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train,y_train)
    
    pred = knn.predict(X_test)
   
    print("\nTask Completed")
    print("----------------------------------------------")
    printRes(k, y_test, pred)
    
    kPlot(X_train, X_test, y_test, y_train) 
    
    makePred(X, knn)
    
    res, k = setK()
    if(res):
        knn = KNeighborsClassifier(n_neighbors = k)
        knn.fit(X_train,y_train)
        pred = knn.predict(X_test)
        printRes(k, y_test, pred)
        
        makePred(X, knn)
    


# In[15]:


KNN()


# In[ ]:




