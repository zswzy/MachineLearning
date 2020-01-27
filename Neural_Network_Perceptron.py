# Import necessary packages
print('importing packages...')
import numpy as np
import pandas as pd
from random import random
from scipy.optimize import fmin_l_bfgs_b
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
print('\nimporting finished')

#reading data
print('\nreading data...')
Data = pd.read_csv('train_V2.csv')
Data.head(10)
Data=Data.round({'winPlacePerc':2})
Data = Data.dropna(axis = 0)
data = Data.drop(['Id','groupId','matchId'],axis = 1)
data = data[(Data['matchType'] == ('solo' or 'solo-fpp'))]
data = data.drop(['matchType'],axis = 1) #最终清洗之后的数据集
print('    reading finished')

print('\nprocessing data...')
X_train,X_test,Y_train,Y_test=train_test_split(data.drop(['winPlacePerc'],axis = 1),data['winPlacePerc'],test_size=0.4)

X_Train=X_train.values
Y_Train=Y_train.values*100

scalar=StandardScaler()
scalar.fit(X_Train)
X_Train=scalar.transform(X_Train)
scalar=StandardScaler()
scalar.fit(X_test.values)
X_Test=scalar.transform(X_test.values)
print('    processing finished')


# unroll_parameters, transform a list of weight matrices into an 1D array
def unroll_params(Theta):
    nn_params = np.reshape(Theta[0], (1, -1)).transpose()
    for i in range(1, len(Theta)):
        nn_params = np.concatenate((nn_params, np.reshape(Theta[i], (1, -1)).transpose()))

    nn_params = np.ndarray.flatten(nn_params)

    return nn_params

def roll_params(nn_params, layers):
    # Setup some useful variables
    num_layers = len(layers)
    Theta = []
    index = 0
    for i in range(num_layers - 1):
        step = layers[i + 1] * (layers[i] + 1)
        Theta.append(np.reshape(nn_params[index:(index + step)], (layers[i + 1], (layers[i] + 1))))

        index = index + step

    return Theta


# randomly initialyze the weights
def randInitializeWeights(layers):
    num_of_layers = len(layers)
    epsilon = 1

    Theta = []
    for i in range(num_of_layers - 1):
        W = np.zeros((layers[i + 1], layers[i] + 1), dtype='float64')
        for m in range(len(W)):
            for n in range(len(W[0])):
                W[m][n] = 2 * epsilon * random() - epsilon
        Theta.append(W)

    return Theta

# Regularized CostFunction of the neural network
def costFunction(nn_weights, layers, X, y, num_labels, l):
    # Setup some useful variables
    m = X.shape[0]
    num_layers = len(layers)

    # Unroll Params
    Theta = roll_params(nn_weights, layers)
    J = 0
    yv = np.zeros((num_labels, m))
    for i in range(m):
        yv[int(y[i]), i] = 1.0

    # feedforward
    activation = np.transpose(np.concatenate((np.ones((m, 1)), X), axis=1))
    activations = [activation]
    zs = []  # list to store all the z vectors, layer by layer
    for i in range(num_layers - 1):
        z = np.dot(Theta[i], activation)
        zs.append(z)
        if i == (num_layers - 2):  # Final layer
            activation = sigmoid(z)
        else:
            activation = np.concatenate((np.ones((1, m)), sigmoid(z)), axis=0)

        activations.append(activation)
        # Cost Function
    J = (1.0 / m) * (np.sum(-1.0 * yv * np.log(activations[-1]) - (1.0 - yv) * np.log(1.0 - activations[-1])))
    for i in range(num_layers - 1):
        J = J + (l / (2.0 * m)) * np.sum(pow(Theta[i][:, 1:], 2.0))

    return J


# sigmoid function
def sigmoid(z):

    # SIGMOID returns sigmoid function evaluated at z
    g = np.zeros(np.shape(z))
    g=1/(1+np.exp(-z))
    return g


# sigmoid gradient
def sigmoidGradient(z):

    g=np.ones(np.shape(z))
    #calculated the gradient for all shapes of z
    if np.shape(z)==():
        g=sigmoid(z)*(1-sigmoid(z))
    elif np.size(z)==len(z):
        for i in range (len(z)):
            g[i]=sigmoid(z[i])*(1-sigmoid(z[i]))
    else:
        for i in range (len(z)):
            for j in range (len(z[0])):
                g[i][j]=sigmoid(z[i][j])*(1-sigmoid(z[i][j]))
    return g

def backwards(nn_weights, layers, X, y, num_labels, lambd):
    # nn_weights: Neural network parameters (vector)
    # layers: a list with the number of units per layer.
    # X: a matrix where every row is a training example for a handwritten digit image
    # y: a vector with the labels of each instance
    # num_labels: the number of units in the output layer
    # lambd: regularization factor
    # Setup some useful variables
    m = X.shape[0]
    num_layers = len(layers)

    # Roll Params
    # The parameters for the neural network are "unrolled" into the vector
    # nn_params and need to be converted back into the weight matrices.
    Theta = roll_params(nn_weights, layers)

    # You need to return the following variables correctly
    Theta_grad = [np.zeros(w.shape) for w in Theta]

    yv = np.zeros((num_labels, m))
    for i in range(m):
        yv[int(y[i]), i] = 1

    activation = np.transpose(np.concatenate((np.ones((m, 1)), X), axis=1))
    activations = [activation]
    zs = []  # list to store all the z vectors, layer by layer
    for i in range(num_layers - 1):
        z = np.dot(Theta[i], activation)
        zs.append(z)
        if i == (num_layers - 2):  # Final layer
            activation = sigmoid(z)
        else:
            activation = np.concatenate((np.ones((1, m)), sigmoid(z)), axis=0)

        activations.append(activation)

    # backward pass
    delta = activations[-1] - yv
    Theta_grad[-1] = (1.0 / m) * np.dot(delta, activations[-2].transpose())
    Theta_grad[-1][:, 1:] = Theta_grad[-1][:, 1:] + (lambd / m) * Theta[-1][:, 1:]
    for l in range(2, num_layers):
        delta = np.dot(Theta[-l + 1].transpose(), delta)[1:, :] * sigmoidGradient(zs[-l])
        Theta_grad[-l] = (1.0 / m) * np.dot(delta, activations[-l - 1].transpose())
        Theta_grad[-l][:, 1:] = Theta_grad[-l][:, 1:] + (lambd / m) * (Theta[-l][:, 1:])

    # Unroll Params
    Theta_grad = unroll_params(Theta_grad)

    return Theta_grad


#Use the weights to predict the output of the neural network
def predict(Theta, X):

    # Useful values
    m = X.shape[0]
    num_labels = Theta[-1].shape[0]
    num_layers = len(Theta) + 1

    p = np.zeros((1, m))
    h = X
    activation = np.transpose(np.concatenate((np.ones((m, 1)), X), axis=1))
    for i in range(num_layers - 1):
        z = np.dot(Theta[i], activation)
        if i == (num_layers - 2):
            activation = sigmoid(z)
        else:
            activation = np.concatenate((np.ones((1, m)), sigmoid(z)), axis=0)
    p = np.argmax(activation, axis=0)

    return (p/100.0)

def train_NN(layers,X_Train,Y_Train,X_Test):

    print("\nSetting up Neural Network Structure ...\n")

    input_layer_size = len(X_Train[0])
    num_labels = 101  # 101 labels, from 0 to 1.00

    print("\nInitializing Neural Network Parameters ...\n")
    Theta = randInitializeWeights(layers)
    # Unroll parameters
    nn_weights = unroll_params(Theta)

    print("\nTraining Neural Network... \n" )

    # We can change the regularized factor
    lambd = 0
    #train the model
    res = fmin_l_bfgs_b(costFunction, nn_weights, fprime=backwards, args=(layers, X_Train, Y_Train, num_labels, lambd),
                        maxfun=50, factr=1., disp=True)
    Theta = roll_params(res[0], layers)

    #nn_weights = unroll_params(Theta)

    print("\nPredictinf results... \n")

    pred = predict(Theta, X_test)
    return pred

#main
def main():
    layers=[24,60,300,101]
    pred=train_NN(layers,X_Train,Y_Train,X_Test)
    Y_pred=[]
    for i in range (len(pred)):
        Y_pred.append(float(pred[i]))
    print('The mean squared error is : %s'%(mean_squared_error(Y_test,Y_pred)))

main()