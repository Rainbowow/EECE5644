import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

#data generated from matlab
def DATA_PROCESS():
    X_train=np.loadtxt('q2_X_train.txt',dtype=np.float32,delimiter=',') #(2,1000)
    y_train=np.loadtxt('q2_y_train.txt',dtype=np.float32,delimiter=',')
    X_test=np.loadtxt('q2_X_test.txt',dtype=np.float32,delimiter=',') #(2,10000)
    y_test=np.loadtxt('q2_y_test.txt',dtype=np.float32,delimiter=',')

    temp=[]
    for i in range(0,X_train.shape[1]):
        temp.append(X_train[0,i])
        temp.append(X_train[1,i])
    X_train=np.array(temp).reshape(-1,2) #(1000,2)

    temp1=[]
    for i in range(0,X_test.shape[1]):
        temp1.append(X_test[0,i])
        temp1.append(X_test[1,i])
    X_test=np.array(temp1).reshape(-1,2) #(10000,2)
    return X_train,y_train,X_test,y_test

X_train,y_train,X_test,y_test=DATA_PROCESS()

def GRID_SEARCH_SVM(X,y):
    best_score = 0
    for gamma in [0.001,0.01,0.1,1,10,100]:
        for C in [0.001,0.01,0.1,1,10,100]:
            svc=SVC(kernel='rbf',C=C,gamma=gamma)
            scores = cross_val_score(svc,X,y,cv=10)#10-fold cross-validation
            score = scores.mean()
            if score > best_score:
                best_score = score
                best_parameters = {'gamma':gamma,'C':C}
    print("Best score on validation set:{:.2f}".format(best_score))
    print("Best parameters:{}".format(best_parameters))


#GRID_SEARCH_SVM(X_train,y_train)

def TRAIN_FINAL_SVM(X,y,C,gamma):
    svc=SVC(kernel='rbf',C=C,gamma=gamma).fit(X,y)
    # test_score = svc.score(X_test,y_test)
    # return test_score

    # Plotting decision regions
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X_test[:, 0], X_test[:, 1], s=3,c=y_test,marker='.')
    plt.show()


TRAIN_FINAL_SVM(X_train,y_train,100,0.1)