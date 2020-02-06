import numpy as np
import pandas as pd
import scipy
## Load Data
X_train_path = 'X_train'
X_test_path = 'X_test'
Y_train_path = 'Y_train'

X_train = np.genfromtxt(X_train_path,delimiter = ',',skip_header = 1)
Y_train = np.genfromtxt(Y_train_path,delimiter = ',',skip_header = 1)

def _sigmoid(z):
    return np.clip(1/(1+np.exp(-z)),1e-6,1-(1e-6))
def get_prob(X,w,b):
    return _sigmoid(np.add(np.matmul(X,w),b))
def infer(X,w,b):
    return get_prob(X,w,b)
def _cross_entropy(y_pred,Y_label):
    cross_entropy = -np.dot(Y_label,np.log(y_pred))-np.dot((1-Y_label),np.log(1-y_pred))
    return cross_entropy
def _gradient(X,Y_label,w,b): ##return the mean of gradient of the cross entropy
    y_pred = get_prob(X,w,b)
    pred_error = Y_label - y_pred
    w_grad = -np.mean(np.multiply(pred_error.T,X.T),1)
    b_grad = -np.mean(pred_error)
    return w_grad,b_grad

def _gradient_regularization(X,Y_label,w,b,lamda):
    y_pred = get_prob(X, w, b)
    pred_error = Y_label - y_pred
    w_grad = -np.mean(np.multiply(pred_error.T, X.T), 1)+lamda*w
    b_grad = -np.mean(pred_error)
    return w_grad, b_grad
def 

