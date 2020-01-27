print('importing packages...')
import numpy as np
import pandas as pd
from random import random
from scipy.optimize import fmin_l_bfgs_b
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
print('\nimporting finished')
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

#reading data
print('\nreading data...')
Data = pd.read_csv('train_V2.csv')
Data.head(10)
#Data=Data.round({'winPlacePerc':2})
Data = Data.dropna(axis = 0)
data = Data.drop(['Id','groupId','matchId'],axis = 1)
data = data[(Data['matchType'] == ('solo' or 'solo-fpp'))]
data = data.drop(['matchType'],axis = 1) #最终清洗之后的数据集
print('    reading finished')

print('\nprocessing data...')
X_train,X_test,Y_train,Y_test=train_test_split(data.drop(['winPlacePerc'],axis = 1),data['winPlacePerc'],test_size=0.4)

X_Train=X_train.values
Y_Train=Y_train.values

scalar=StandardScaler()
scalar.fit(X_Train)
X_Train=scalar.transform(X_Train)
scalar=StandardScaler()
scalar.fit(X_test.values)
X_Test=scalar.transform(X_test.values)
print('    processing finished')

print('\ntraining model...')
clf = MLPRegressor(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(100,40), random_state=1)
print('\n    training finished')

print('\npredicting the results...')
clf.fit(X_Train,Y_Train)
pred=clf.predict(X_Test)
print('\npredicion finished')

print('The mean squared error is : %s'%(mean_squared_error(Y_test.values,pred)))