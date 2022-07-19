# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 22:26:42 2022

@author: sayan
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

y=y.reshape(len(y),1)

#print(y)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

#print(X)
#print(y)

#Training the SVR model in the data set
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)


#predicting the new value                
sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1,1))  #in the range of feature scaling. need to reverse

#visualising the SVR results(for higher definition and smoother curve)
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))

plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y),color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1, 1)), color = 'blue' )
plt.title('Truth or bluff (Support Vector Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()