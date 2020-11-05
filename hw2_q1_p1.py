import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import homework2
from homework2 import plot_boundary,Plot_data,Generate_and_plot_data,Roc_plot,find_theta

plt.rcParams['figure.figsize']=9,7

#generate data
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
    x_0=[i for i in range(0,samples) if labels[0,i]==0]
    x_1=[i for i in range(0,samples) if labels[0,i]==1]

    plt.scatter(X[0,x_0],X[1,x_0],s=5,color='r',label='class 0',marker='*')
    plt.scatter(X[0,x_1],X[1,x_1],s=2,color='b',label='class 1',marker='o')
    plt.title('Actual data distribution')
    plt.xlabel('feature x1')
    plt.ylabel('feature x2')
    plt.legend()
    plt.show()

def Generate_and_plot_data(samples,priors,mean_0,cov_0,mean_1,cov_1,w1,w2):
    #labels
    labels=np.zeros((2,samples))
    labels[0,:]=(np.random.uniform(0,1,samples) >= priors[0]).astype(float)
    #data
    X=np.zeros((2,samples))
    for i in range(0,samples):
        if(labels[0,i]==0):
            X[:,i]=w1*np.random.multivariate_normal(mean_0[:,0],cov_0[:,0],1)+w2*np.random.multivariate_normal(mean_0[:,1],cov_0[:,1],1)
        else:
            X[:,i]=np.random.multivariate_normal(mean_1,cov_1,1)
    
    Plot_data(X,samples,labels)
    return X,labels

def Roc_plot(X,labels,samples,mean_0,cov_0,mean_1,cov_1):
    class_0_count=float(list(labels[0,:]).count(0))
    class_1_count=float(list(labels[0,:]).count(1))

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
        x00 = [i for i in range(labels.shape[1]) if (labels[1,i] == 0 and labels[0,i] == 0)]
        x01 = [i for i in range(labels.shape[1]) if (labels[1,i] == 0 and labels[0,i] == 1)]
        x10 = [i for i in range(labels.shape[1]) if (labels[1,i] == 1 and labels[0,i] == 0)]
        x11 = [i for i in range(labels.shape[1]) if (labels[1,i] == 1 and labels[0,i] == 1)]
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
    

def find_theta(train_data,alpha,iterations,train_labels,test_data, type_ = 'l'):
    if(type_ == 'l'):
        z = np.c_[np.ones((train_data.shape[1])),train_data.T].T
        w = np.zeros((3,1))
    else:
        z = np.c_[np.ones((train_data.shape[1])),train_data[0],train_data[1],train_data[0]*train_data[0],train_data[0]*train_data[1], train_data[1]*train_data[1]].T
        w = np.zeros((6,1))

    for i in range(iterations):
        h = 1/ (1+ np.exp(-(np.dot(w.T,z))))
        cost_gradient = (1/float(z.shape[1])) * np.dot(z,(h-train_labels[0]).T)
        w = w - alpha*cost_gradient

    if(type_ == 'l'):
        z = np.c_[np.ones((test_data.shape[1])),test_data.T].T
    else:
        z = np.c_[np.ones((test_data.shape[1])),test_data[0],test_data[1],test_data[0]*test_data[0],test_data[0]*test_data[1], test_data[1]*test_data[1]].T
    decisions = np.zeros((1,test_data.shape[1]))
    h = 1/ (1+np.exp(-(np.dot(w.T,z))))
    decisions[0,:] = (h[0,:]>=0.5).astype(int)
    return (w,decisions)

def plot_boundary(labels,dataset,w):
    X = dataset
    w_10=w


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
    dsg=np.zeros((100,100))
    a=np.array(np.meshgrid(horizontalGrid,verticalGrid))
    for i in range(100):
        for j in range(100):
            x1 = a[0][i][j]
            x2 = a[1][i][j]
            z = np.c_[1,x1,x2].T
            dsg[i][j] = np.sum(np.dot(w_10.T,z))
    plt.contour(a[0],a[1],dsg, levels = [0])
    plt.show()

np.random.seed(7)

D_train_100,labels_100=Generate_and_plot_data(100,priors,m_0,cov_0,m_1,cov_1,w1,w2)
D_train_1000,labels_1000=Generate_and_plot_data(1000,priors,m_0,cov_0,m_1,cov_1,w1,w2)
D_train_10000,labels_10000=Generate_and_plot_data(10000,priors,m_0,cov_0,m_1,cov_1,w1,w2)
D_test_20000,labels_20000=Generate_and_plot_data(20000,priors,m_0,cov_0,m_1,cov_1,w1,w2)

#D_test_1000,test_labels_1000=Generate_and_plot_data(1000,priors,m_0,cov_0,m_1,cov_1,w1,w2)
#print(D_train_100,labels_100)

#Roc_plot(D_train_20000,labels_20000,20000,m_0,cov_0,m_1,cov_1)
#Roc_plot(D_train_100,labels_100,100,m_0,cov_0,m_1,cov_1)
w_10000,decisions_10000 = find_theta(D_train_10000,0.01,2000,labels_10000,D_test_20000, type_='l')
#w_1000, decisions_1000 = find_theta(D_train_1000,0.05,2000,labels_1000,test_labels_1000, type_='l')

for decisions in [decisions_10000]:
    x00 = [i for i in range(20000) if (labels_20000[0,i] == 0 and decisions[0,i] == 0)]
    x11 = [i for i in range(20000) if (labels_20000[0,i] == 1 and decisions[0,i] == 1)]
    print(100 - ((len(x00)+len(x11))/100.0))

plot_boundary(np.vstack((labels_20000[0,:],decisions_10000)),D_test_20000,w_10000)