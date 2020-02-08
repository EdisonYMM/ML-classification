import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
## Load Data
X_train_path = 'X_train'
X_test_path = 'X_test'
Y_train_path = 'Y_train'
output_path = 'output2.csv'

X_train = np.genfromtxt(X_train_path, delimiter = ',', skip_header = 1)
Y_train = np.genfromtxt(Y_train_path, delimiter = ',', skip_header = 1)
## activation function
def _sigmoid(z):
    return np.clip(1/(1+np.exp(-z)), 1e-6, 1-(1e-6))
def get_prob(X, w, b):
    return _sigmoid(np.add(np.matmul(X, w), b))
def infer(X, w, b):
    return np.round(get_prob(X, w, b))
def _cross_entropy(y_pred, Y_label):
    cross_entropy = -np.dot(Y_label, np.log(y_pred))-np.dot((1-Y_label), np.log(1-y_pred))
    return cross_entropy
def _gradient(X, Y_label, w, b): ##return the mean of gradient of the cross entropy
    y_pred = get_prob(X, w, b)
    pred_error = Y_label - y_pred
    w_grad = -np.mean(np.multiply(pred_error.T,X.T),1)
    b_grad = -np.mean(pred_error)
    return w_grad,b_grad
## 正则化，可以使曲线更加平滑
def _gradient_regularization(X,Y_label,w,b,lamda):
    y_pred = get_prob(X, w, b)
    pred_error = Y_label - y_pred
    w_grad = -np.mean(np.multiply(pred_error.T, X.T), 1)+lamda*w
    b_grad = -np.mean(pred_error)
    return w_grad, b_grad
def _loss(y_pred,Y_label,lamda,w):
    return _cross_entropy(y_pred,Y_label) + lamda*np.sum(np.square(w))
## 计算正确率
def accuracy(Y_pred,Y_label):
    acc = np.sum(Y_pred == Y_label)/len(Y_pred)
    return acc
## Data normalization, 将所有的输入数据都转化为0-1的数据
def _normalize_column_0_1(X,train=True,specified_column = None,X_min = None,X_max = None):
    if train:
        if specified_column == None:
            specified_column = np.arange(X.shape[1])
        length = len(specified_column)
        X_max = np.reshape(np.max(X[:,specified_column],0),(1,length)) ##找到每列的最大值
        X_min = np.reshape(np.min(X[:,specified_column],0),(1,length)) ##找到每列的最小值

    X[:,specified_column] = np.divide(np.subtract(X[:,specified_column],X_min),np.subtract(X_max,X_min))
    return X,X_max,X_min


def _normalize_column_normal(X, train=True, specified_column=None, X_mean=None, X_std=None):
    if train:
        if specified_column == None:
            specified_column = np.arange(X.shape[1])
        length = len(specified_column)
        X_mean = np.reshape(np.mean(X[:, specified_column], 0), (1, length))
        X_std = np.reshape(np.std(X[:, specified_column], 0), (1, length))
    X[:, specified_column] = np.divide(np.subtract(X[:, specified_column], X_mean), X_std)
    return X, X_mean, X_std

def _shuffle(X,Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize],Y[randomize])
def train_dev_splt(X,Y,dev_size = 0.25):
    train_len = int(round(len(X)*(1-dev_size)))
    return X[0:train_len],Y[0:train_len],X[train_len:None],Y[train_len:None]
## normalize 这些列
col = [0,1,3,4,5,7,10,12,25,26,27,28]
X_train, X_mean, X_std = _normalize_column_normal(X_train, specified_column=col)
## 下面是正式的training，(Logistic Regression)
def train(X_train,Y_train):
    dev_size = 0.1155
    X_train,Y_train,X_dev,Y_dev = train_dev_splt(X_train,Y_train,dev_size)
    ## initiate weights
    w = np.zeros((X_train.shape[1],))
    b = np.zeros((1,))
    ##是否需要regularization？
    regularize = True
    if regularize:
        lamda = 0.001
    else:
        lamda = 0

    max_iter = 40 ## max iteration number
    batch_size = 32 ## batch可以减小bias
    learning_rate = 0.2
    num_train = len(Y_train)
    num_dev = len(Y_dev)
    step = 1

    loss_train = []
    loss_validation = []
    train_acc = []
    dev_acc = []
    for epoch in range(max_iter):
        X_train,Y_train = _shuffle(X_train,Y_train) ##每个iteration都要将数据打乱
        ## Logistic Regression train with batch
        total_loss = 0.0

        for idx in range(int(np.floor(len(Y_train)/batch_size))):
            ## 每次都取一个batch训练
            X = X_train[idx*batch_size:(idx+1)*batch_size]
            Y = Y_train[idx*batch_size:(idx+1)*batch_size]

            w_grad,b_grad = _gradient_regularization(X,Y,w,b,lamda)
            ## learning rate 随着更新的次数的增加而减少
            w = w - learning_rate/np.sqrt(step)*w_grad
            b = b - learning_rate/np.sqrt(step)*b_grad
            ## 更新step
            step+=1
        ## 计算正确率 training set和 validation set
        y_train_pred = get_prob(X_train,w,b)
        Y_train_pred = np.round(y_train_pred)
        train_acc.append(accuracy(Y_train_pred,Y_train))
        loss_train.append(_loss(y_train_pred,Y_train,lamda,w)/num_train)

        y_dev_pred = get_prob(X_dev,w,b)
        Y_dev_pred = np.round(y_dev_pred)
        dev_acc.append(accuracy(Y_dev_pred,Y_dev))
        loss_validation.append(_loss(y_dev_pred,Y_dev,lamda,w)/num_dev)

    return w,b#,loss_train,loss_validation,train_acc,dev_acc
## 对test数据做同样的normalization
X_test = np.genfromtxt(X_test_path,delimiter = ',',skip_header = 1)
X_test,_,_ = _normalize_column_normal(X_test,train = False,specified_column=col,X_mean = X_mean,X_std = X_std)


w,b = train(X_train,Y_train)
result = infer(X_test,w,b)
with open(output_path,'w') as f:
    f.write('id,label\n')
    for i,v in enumerate(result):
        f.write('%d,%d\n'%(i+1,v))

ind = np.argsort(np.abs(w))[::-1]
with open(X_test_path) as f:
    content = f.readline().rstrip('\n')
features = np.array([x for x in content.split(',')])
for i in ind[0:10]:
    print(features[i],w[i])
