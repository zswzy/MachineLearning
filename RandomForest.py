print('importing packages...')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import mean_squared_error
print('    importing finished')

print('\nreading data...')
Data = pd.read_csv('train_V2.csv')
Data=Data.round({'winPlacePerc':2})
Data = Data.dropna(axis = 0)
data = Data.drop(['Id','groupId','matchId'],axis = 1)
data = data[(Data['matchType'] == ('solo' or 'solo-fpp'))]
data = data.drop(['matchType'],axis = 1) #最终清洗之后的数据集
print('    reading finished')

print('\nprocessing data...')
X_train,X_test,Y_train,Y_test=train_test_split(data.drop(['winPlacePerc'],axis = 1),data['winPlacePerc'],test_size=0.1)

XX_train=X_train.values
YY_train=[]
for i in range (len(Y_train.values)):
    YY_train.append(str(Y_train.values[i]))
YY_train=np.array(YY_train)
print('    processing finished')

print('\ntraining models...')
clf = RFC(n_estimators=100,n_jobs=-1,random_state=0)
clf.fit(XX_train,YY_train)
print('    training finished')

print('\nverifying models...')
Y_pred=clf.predict(X_test.values)
YY_pred=[]
for i in range (len(Y_pred)):
    YY_pred.append(float(Y_pred[i]))
print('The mean squared error is : %s'%(mean_squared_error(Y_test,YY_pred)))
print('    verifying finished')