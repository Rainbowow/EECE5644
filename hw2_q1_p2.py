import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_curve,precision_score
#data distribution
features = 2

m_1=np.zeros((1,2))
m_1=[3,2]
#print(m_1)
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
#D_train_100,labels_100=Generate_and_plot_data(100,priors,m_0,cov_0,m_1,cov_1,w1,w2)
#_train_1000,labels_1000=Generate_and_plot_data(1000,priors,m_0,cov_0,m_1,cov_1,w1,w2)
D_train_10000,labels_10000=Generate_and_plot_data(10000,priors,m_0,cov_0,m_1,cov_1,w1,w2)
D_test_20000,labels_20000=Generate_and_plot_data(20000,priors,m_0,cov_0,m_1,cov_1,w1,w2)
#D_train_10000,labels_10000=generate_data(10000,prior=priors)

def Roc_plot(X,labels,samples,mean_0,cov_0,mean_1,cov_1):
    class_0_count=float(list(labels).count(0))
    class_1_count=float(list(labels).count(1))

    fpr=[]
    tpr=[]

    min_Perror=[]

    q=list(sorted(X[0,:]))
    gamma_list=[]
    for i in range(0,samples-1):
        gamma_list.append((q[i]+q[i+1])/2)
        gamma_list=[gamma_list[i] for i in range(0,len(gamma_list)) if gamma_list[i]>=0]

    logValpdf1=np.log(multivariate_normal.pdf(X.T,mean=mean_1,cov=cov_1))
    logValpdf0=np.log(multivariate_normal.pdf(X.T,mean=0.5*mean_0[:,0]+0.5*mean_0[:,1],cov=0.5*cov_0[:,:,0]+0.5*cov_0[:,:,1]))
    discriminant_score=logValpdf1-logValpdf0
        # Code to plot ROC curve
    # Calculate probability of minimum error for all values in gamma_list
    for gamma in gamma_list:
        labels[1,:] = (discriminant_score >= np.log(gamma)).astype(int)
        x00 = [i for i in range(labels.shape[0]) if (labels[1,i] == 0 and labels[0,i] == 0)]
        x01 = [i for i in range(labels.shape[0]) if (labels[1,i] == 0 and labels[0,i] == 1)]
        x10 = [i for i in range(labels.shape[0]) if (labels[1,i] == 1 and labels[0,i] == 0)]
        x11 = [i for i in range(labels.shape[0]) if (labels[1,i] == 1 and labels[0,i] == 1)]
        fpr.append(len(x10)/class_0_count)
        tpr.append(len(x11)/class_1_count)
        min_Perror.append(1 - ((len(x00)+len(x11))/10000.0))

    # Plot the ROC curve
    plt.plot(fpr,tpr,color = 'red' )
    plt.plot(fpr[np.argmin(min_Perror)],tpr[np.argmin(min_Perror)],'*',color = 'black')

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.annotate(s='Min_P(e) = ' + str(round(min_Perror[np.argmin(min_Perror)],3)) + '\n'
    +'TPR = '+ str(round(tpr[np.argmin(min_Perror)],2)) + '\n'
    +'FPR = '+ str(round(fpr[np.argmin(min_Perror)],2)) + '\n'
    +'Gamma = '+ str(round(gamma_list[np.argmin(min_Perror)],2)),xy =(fpr[np.argmin(min_Perror)],tpr[np.argmin(min_Perror)]),
    xytext=(fpr[np.argmin(min_Perror)]+0.3,tpr[np.argmin(min_Perror)]),
    arrowprops=dict(facecolor='black', width = 0.01,headwidth = 5,shrink=0))
    plt.show()
    
    labels[1,:]=(discriminant_score>=np.log(1.5)).astype(int)
    x00=[i for i in range(labels.shape[1]) if (labels[0,i]==0 and labels[1,i]==0)]
    x01=[i for i in range(labels.shape[1]) if (labels[0,i]==0 and labels[1,i]==1)]
    x10=[i for i in range(labels.shape[1]) if (labels[0,i]==1 and labels[1,i]==0)]
    x11=[i for i in range(labels.shape[1]) if (labels[0,i]==1 and labels[1,i]==1)]

    plt.plot(X[0,x00],X[1,x00],'.',color='g',markersize=6)
    plt.plot(X[0,x01],X[1,x01],'.',color='r',markersize=6)
    plt.plot(X[0,x11],X[1,x11],'+',color='g',markersize=6)
    plt.plot(X[0,x10],X[1,x10],'+',color='r',markersize=6)
    plt.legend(["class 0 correctly classified","class 0 wrongly classified","class 1 correctly classified","class 1 wrongly classified"])
    plt.xlabel("feature x1")
    plt.ylabel("feature x2")
    plt.title("Discribution after classification overlapped by decisoin boundaries")

    horizontalGrid=np.linspace(np.floor(min(X[0,:])),np.ceil(max(X[0,:])),100)
    verticalGrid=np.linspace(np.floor(min(X[1,:])),np.ceil(max(X[1,:])),100)
    a = np.array(np.meshgrid(horizontalGrid,verticalGrid))
    dsg = np.zeros((100,100))
    for i in range(100):
        for j in range(100):
            p=multivariate_normal.pdf(np.array([a[0][i][j],a[1][i][j]]),mean=mean_1,cov=cov_1)
            q=multivariate_normal.pdf(np.array([a[0][i][j],a[1][i][j]]),mean=0.5*mean_0[:,0]+0.5*mean_0[:,1],cov=0.5*cov_0[:,:,0]+0.5*cov_0[:,:,1])
            dsg[i][j]=np.log(p)-np.log(q)-np.log(1.5)
    plt.contour(a[0],a[1],dsg)
    plt.show()

#Roc_plot(D_test_20000,labels_20000,20000,mean_0=m_0,cov_0=cov_0,mean_1=m_1,cov_1=cov_1)

def gaussian_mle(data):                                                                                                                                                                               
    mu = data.mean(axis=0)                                                                                                                                                                            
    var = (data-mu).T @ (data-mu) / data.shape[0] #  this is slightly suboptimal, but instructive

    return mu, var

def gaussian_mix_mle(data,label):
    X,Y =data,label
    GMM = GaussianMixture(n_components=2)
    GMM.fit(X)
    print(GMM.means_, GMM.predict_proba(Y))

data_temp_1=[]
data_temp_0=[]
for i in range(20000):
    if(labels_20000[i]==1):
        data_temp_1.append(D_test_20000[i])
    else:
        data_temp_0.append(D_test_20000[i])

data_true=np.array(data_temp_1)
data_false=np.array(data_temp_0)
#data_true=D_train_10000[ i for i in range(10000) if (labels_10000[i]==1)]
# print(data_true.shape)
# print(gaussian_mle(D_train_10000))

def gmm_model(data):
    X=data
    GMM = GaussianMixture(n_components=2)
    GMM.fit(X)
    Y = np.random.randint(-10, 20, size=(1, 2))
    print(GMM.means_, GMM.predict_proba(Y))

#gmm_model(data_temp_0)
GMM = GaussianMixture(n_components=1,covariance_type='diag')
GMM.fit(D_train_10000)
y_pred=GMM.predict(D_test_20000)
print(y_pred)
print(precision_score(labels_20000, y_pred, average='weighted'))