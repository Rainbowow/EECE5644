import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_curve,auc,precision_score
from sklearn import linear_model
#data distribution
features = 2

m_1=np.zeros((1,2))
m_1=[3,2]

cov_1=np.zeros((2,2))
cov_1[:,0]=[2,0]
cov_1[:,1]=[0,2]

m_0=np.zeros((2,2))
m_0[:,0]=[5,0]
m_0[:,1]=[0,4]


cov_0=np.zeros((2,2,2))
cov_0[:,:,0]=[[4,0],[0,2]]
cov_0[:,:,1]=[[1,0],[0,3]]

priors=[0.6,0.4]
w1=0.5
w2=0.5
def Plot_data(X,samples,labels):
    x_0=[i for i in range(0,samples) if labels[i]==0]
    x_1=[i for i in range(0,samples) if labels[i]==1]

    plt.scatter(X[x_0,0],X[x_0,1],s=5,color='r',label='class 0',marker='*')
    plt.scatter(X[x_1,0],X[x_1,1],s=2,color='b',label='class 1',marker='o')
    plt.title('Actual data distribution')
    plt.xlabel('feature x1')
    plt.ylabel('feature x2')
    plt.legend()
    plt.show()

def Generate_and_plot_data(samples,priors,mean_0,cov_0,mean_1,cov_1,w1,w2):
    #labels
    labels=(np.random.uniform(0,1,samples) >= priors[0]).astype(float)
    #data
    X=np.zeros((samples,2))
    for i in range(0,samples):
        if(labels[i]==0):
            X[i,:]=w1*np.random.multivariate_normal(mean_0[:,0],cov_0[:,0],1)+w2*np.random.multivariate_normal(mean_0[:,1],cov_0[:,1],1)
        else:
            X[i,:]=np.random.multivariate_normal(mean_1,cov_1,1)
    
    Plot_data(X,samples,labels)
    return X,labels

np.random.seed(7)
D_train_100,labels_100=Generate_and_plot_data(100,priors,m_0,cov_0,m_1,cov_1,w1,w2)
#D_train_1000,labels_1000=Generate_and_plot_data(1000,priors,m_0,cov_0,m_1,cov_1,w1,w2)
#D_train_10000,labels_10000=Generate_and_plot_data(10000,priors,m_0,cov_0,m_1,cov_1,w1,w2)
#D_train_20000,labels_20000=Generate_and_plot_data(20000,priors,m_0,cov_0,m_1,cov_1,w1,w2)

D_test_20000,labels_20000=Generate_and_plot_data(20000,priors,m_0,cov_0,m_1,cov_1,w1,w2)
#D_train_10000,labels_10000=generate_data(10000,prior=priors)

data_temp_1=[]
data_temp_0=[]
label_temp_0=[]
label_temp_1=[]
for i in range(100):
    if(labels_100[i]==1):
        data_temp_1.append(D_train_100[i])
        label_temp_1.append(labels_100[i])
    else:
        data_temp_0.append(D_train_100[i])
        label_temp_0.append(labels_100[i])

data_true=np.array(data_temp_1)
data_false=np.array(data_temp_0)
label_true=np.array(label_temp_1)
label_false=np.array(label_temp_0)

logistic=linear_model.LogisticRegression()
logistic.fit(D_train_100,labels_100)
y_pred=logistic.predict(D_test_20000)
print(precision_score(labels_20000, y_pred, average='weighted'))

# y_score = logistic.fit(D_train_100,labels_100).decision_function(D_test_20000)

# fpr,tpr,threshold = roc_curve(labels_20000, y_score) ###计算真正率和假正率
# roc_auc = auc(fpr,tpr) ###计算auc的值
# print(threshold)
# plt.figure()
# lw = 2
# plt.figure(figsize=(10,10))
# plt.plot(fpr, tpr, color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC curve')
# plt.legend(loc="lower right")
# plt.show()