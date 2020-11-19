import numpy as np
import matplotlib.pyplot as plt
import torchvision
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import RidgeCV,Ridge
from sklearn.metrics import log_loss,r2_score
#Generate data
random_a=np.arange(1,8).reshape(1,-1)#1 2 3 4 5 6 7 


mu=[10,9,8,7,6,5,4]
#rand a cov
# A = np.random.randint(0,high=2,size=(7,7))
# B = np.dot(A,A.transpose())
# cov = B+B.T # makesure symmetric
cov=[[ 6 , 0,  4 , 0  ,4  ,6  ,2],
    [ 0 , 4 ,0  ,4  ,4 , 2 , 0],
    [ 4 , 0 , 4 , 0 , 4 , 4 , 0],
    [ 0 , 4 , 0 , 6 , 4 , 2 , 0],
    [ 4 , 4 , 4 , 4 ,10 , 6 , 0],
    [ 6 , 2 , 4 , 2 , 6 , 8 , 2],
    [ 2 , 0 , 0  ,0 , 0 ,2 , 2]]
#print(cov)
def GENERATE_DATA(sample_size,a):

    #data
    data = np.random.multivariate_normal(mu,cov,sample_size).astype('double')
    #label
    I=np.zeros((7,7))
    for i in range(7):
        I[i,i]=1
    z = np.random.multivariate_normal(np.zeros(7),a*I,sample_size)
    temp_label = []
    for i in range(sample_size):
        temp=data[i,:]+z[i,:]
        temp=np.dot(random_a,temp)
        temp=temp+np.random.randn(1)#equal to np.random.normal(loc=0, scale=1, 1)
        temp_label.append(temp)
    label=np.array(temp_label).astype('double')
    return data,label

# D_train_100,train_labels_100=GENERATE_DATA(100,1)
# D_test_10000,test_labels_10000=GENERATE_DATA(10000,1)
a_noise=[1e-3*np.trace(cov)/7, 1e-2*np.trace(cov)/7, 1e-1*np.trace(cov)/7, 1*np.trace(cov)/7,1e2*np.trace(cov)/7,1e3*np.trace(cov)/7]
def TRAIN_AND_VALIDATE(a):
    a_s=[]
    r2=[]
    coefs=[]

    for i in a:
        X,y=GENERATE_DATA(100,i)
        D_test_10000,test_labels_10000=GENERATE_DATA(10000,i)
        clf = RidgeCV(alphas=[0.01, 0.1, 0.5, 1, 3, 5, 7, 10, 20, 100],cv=5).fit(X, y)
        #y_pred=clf.predict(D_test_10000)
        
        a_s.append(clf.alpha_)
        #print(clf.alpha_)
        #print(r2_score(test_labels_10000,y_pred))
        best_model=Ridge(alpha=clf.alpha_).fit(X,y)
        y_pred=best_model.predict(D_test_10000)
        r2.append(r2_score(test_labels_10000,y_pred))
        print(best_model.coef_)
        coefs.append(best_model.coef_)
        #print(r2_score(test_labels_10000,y_pred))
    # ax = plt.gca()
    # ax.plot(a_noise, a_s)
    
    # #ax.plot(a_noise,coefs[0])
    
    
    # ax.set_xscale('log')
    
    # ax.set_xlim(ax.get_xlim()[::-1]) 
    # plt.xlabel('a_noise')
    # plt.ylabel('hyper parameters alpha')
    # plt.title('relation between noise and hyper parameters')
    # plt.axis('tight')
    # plt.show()
    
    bx=plt.gca()
    bx.plot(a_noise,r2)
    bx.set_xlim(bx.get_xlim()[::-1]) 
    plt.xlabel('a_noise')
    plt.ylabel('r2 score of each dataset')
    plt.title('relation between noise and r2 score')
    plt.axis('tight')
    plt.show()

    


TRAIN_AND_VALIDATE(a_noise)
