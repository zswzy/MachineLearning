import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#reading data
print('\nreading data...')
Data = pd.read_csv('train_V2.csv')
print('Data size:',Data.shape)
print(Data.head(10))


# eliminate usless data and extract solo mode for training
print('\nstarting eliminating usless data and extract solo mode for training...')
Data = Data.dropna(axis = 0)
data = Data.drop(['Id','groupId','matchId'],axis = 1)
data = data[(Data['matchType'] == ('solo' or 'solo-fpp'))]
data = data.drop(['matchType'],axis = 1)
print('data_size:',data.shape)


# Cross validation and normalize data
print('\nStarting normalizing data and cross validation...')
train_size = int(len(data)*0.9)
X_train = data.drop(['winPlacePerc'],axis = 1)[0:train_size]
X_test = data.drop(['winPlacePerc'],axis = 1)[train_size:]
y_train = data['winPlacePerc'][0:train_size]
y_test = data['winPlacePerc'][train_size:]
m_train = X_train.mean(axis = 0)
m_test = X_test.mean(axis = 0)
X0_train = X_train - m_train
X0_test = X_test - m_test
print('normalized train set shape:',X0_train.shape)
print('normalized test set shape:',X0_test.shape)

# Dimension reduction with SVD methode
print('\nstarting dimension reduction with SVD methode...')
U,S,Vh=np.linalg.svd(X0_train, full_matrices = False)
V = Vh.transpose()
print("Shape U:", U.shape)
print("Shape S:", S.shape)
print("Shape V:", V.shape)

# Chosse the appropriate dimensions that keep 90% of the energy
print("\nstarting dimension reduction with 90% energy...")
threshold= np.sum(S**2) *0.9
index=0
while True :
    if np.sum(S[:index]**2)>=threshold:
        break
    else:
        index += 1
#index = 20 #这里可以手动调整特征值数目
Sigma=S[:index]
Sigma_new=np.diag(S[:index])
U_new=U[:,:index]
V_new=V[:index,:]
Xnew_train=np.dot(X0_train,V_new.T)
#print('The original singular values:',S)
print('The original energy:',np.dot(S,S))
print('We choose %d first singular value.' %(index))
print('new energy:',np.dot(Sigma,Sigma))
#print('Sigma_new:',Sigma_new)
print('Shape of Xnew_train:',Xnew_train.shape)

#Linear Regression
print('\nstarting linear regression...')
Xnew_test = np.dot(X0_test,V_new.T)
reg = LinearRegression().fit(Xnew_train, y_train)
y_train_predict = reg.predict(Xnew_train)
y_test_predict = reg.predict(Xnew_test)
MSE_train = mean_squared_error(y_train,y_train_predict)
MSE_test = mean_squared_error(y_test,y_test_predict)
print('coef:',reg.coef_,'intercept:',reg.intercept_)
print('MSE_train:',MSE_train,'MSE_test:',MSE_test)