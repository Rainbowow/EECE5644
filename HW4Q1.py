import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score



#data generated from matlab
X_train=np.loadtxt('X_train.txt',dtype=np.float32,delimiter=',').reshape(-1,1)
y_train=np.loadtxt('y_train.txt',dtype=np.float32,delimiter=',').reshape(-1,1)
X_test=np.loadtxt('X_test.txt',dtype=np.float32,delimiter=',').reshape(-1,1)
y_test=np.loadtxt('y_test.txt',dtype=np.float32,delimiter=',').reshape(-1,1)



#sklearn model
def TRAIN_AND_VALIDATE(X,y,P):
    net=MLPRegressor(hidden_layer_sizes=(P,1),activation='relu',solver='adam',learning_rate_init=0.1,max_iter=1000)
    train_result=cross_val_score(net, X.astype(np.float32), y.astype(np.int64),cv=10,scoring='neg_mean_squared_error')
    return train_result
    

#print(TRAIN_AND_VALIDATE(X_train,y_train,100))
def GRIDSEARCH():
    mean_error=[]
    for P in range(1,100):
        print(P)
        mean_error.append(sum(TRAIN_AND_VALIDATE(X_train,y_train,P))/10)
    return mean_error

def PLOT_RESULT():
    y=GRIDSEARCH()
    print(np.argmax(y))
    x=range(1,len(y)+1)
    plt.scatter(x,y)
    plt.xlabel('perceptions')
    plt.ylabel('mean_squared_errors')
    plt.show()

#PLOT_RESULT()
def FINAL_MODEL():
    net=MLPRegressor(hidden_layer_sizes=(65,1),activation='relu',solver='adam',learning_rate_init=0.1,max_iter=10000).fit(X_train,y_train)
    y_pred=net.predict(X_test)
    
    mse=mean_squared_error(y_test, y_pred)
    
    plt.figure()
    plt.xticks(range(0,36))
    plt.yticks(range(0,15))
    plt.scatter(X_test,y_test,linewidths=0.1,marker='.')
    plt.scatter(X_test,y_pred,color='red',marker='.')
    plt.text(20,12,'mse=%.4f'%mse)
    plt.show()

FINAL_MODEL()