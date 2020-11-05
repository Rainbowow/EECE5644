import numpy as np
from homework2 import plot_boundary,Plot_data,Generate_and_plot_data,Roc_plot,find_theta
from sklearn.model_selection import KFold
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles

import itertools
from scipy import linalg


#generate data
features = 2


mean = np.zeros((10,2))
for i in range(10):
    mean[i,:]=[0,i+1]


cov = np.zeros((10,2,2))
for i in range(10):
    cov[i,:,:]=np.array([[i,0],[0,i]])

prior = np.array([0.2]*5)

def gen_gmm(samples):
    # seed to recreate the same results everytime

    label = np.zeros(samples)
    a = np.random.uniform(0,1,samples)
    for i in range(0,samples):
        if(a[i] <=prior[0]):
            label[i] = 0
        elif((a[i] >prior[0]) and (a[i] <=prior[0]+prior[1])):
            label[i] = 1
        elif((a[i] >prior[0] +prior[1]) and (a[i] <=prior[0]+prior[1]+prior[2])):
            label[i] = 2
        elif((a[i] >prior[0]+prior[1]+prior[2]) and (a[i] <=prior[0]+prior[1]+prior[2]+prior[3])):
            label[i] = 3
        elif((a[i] >prior[0] +prior[1]+prior[2]+prior[3]) and (a[i] <=prior[0]+prior[1]+prior[2]+prior[3]+prior[4])):
            label[i] = 4
        elif((a[i] >prior[0] +prior[1]+prior[2]+prior[3]+prior[4]) and (a[i] <=prior[0]+prior[1]+prior[2]+prior[3]+prior[4]+prior[5])):
            label[i] = 5
        elif((a[i] >prior[0] +prior[1]+prior[2]+prior[3]+prior[4]+prior[5]) and (a[i] <=prior[0]+prior[1]+prior[2]+prior[3]+prior[4]+prior[5]+prior[6])):
            label[i] = 6
        elif((a[i] >prior[0] +prior[1]+prior[2]+prior[3]+prior[4]+prior[5] +prior[6]) and (a[i] <=prior[0]+prior[1]+prior[2]+prior[3]+prior[4]+prior[5]+prior[6]+prior[7])):
            label[i] = 7
        elif((a[i] >prior[0] +prior[1]+prior[2]+prior[3]+prior[4]+prior[5] +prior[6] +prior[7]) and (a[i] <=prior[0]+prior[1]+prior[2]+prior[3]+prior[4]+prior[5]+prior[6]+prior[7]+prior[8])):
            label[i] = 8
        else:
            label[i] = 9

    X = np.zeros((samples,features))

    for index in range(samples):
        if(label[index] == 0):
            X[index,:] = np.random.multivariate_normal(mean[0,:],cov[0,:,:],1)
        elif(label[index] == 1):
            X[index,:] = np.random.multivariate_normal(mean[1,:],cov[1,:,:],1)
        elif(label[index] == 2):
            X[index,:] = np.random.multivariate_normal(mean[2,:],cov[2,:,:],1)
        elif(label[index] == 3):
            X[index,:] = np.random.multivariate_normal(mean[3,:],cov[3,:,:],1)
        elif(label[index] == 4):
            X[index,:] = np.random.multivariate_normal(mean[4,:],cov[4,:,:],1)
        elif(label[index] == 5):
            X[index,:] = np.random.multivariate_normal(mean[5,:],cov[5,:,:],1)
        elif(label[index] == 6):
            X[index,:] = np.random.multivariate_normal(mean[6,:],cov[6,:,:],1)
        elif(label[index] == 7):
            X[index,:] = np.random.multivariate_normal(mean[7,:],cov[7,:,:],1)
        elif(label[index] == 8):
            X[index,:] = np.random.multivariate_normal(mean[8,:],cov[8,:,:],1)
        else:
            X[index,:] = np.random.multivariate_normal(mean[9,:],cov[9,:,:],1)
    return X,label


def calculate_ML(data, mean_gmm, cov_gmm, al_gmm, model_order):
    cum_sum = np.zeros((data.shape[1]))
    for i in range(model_order):
        cum_sum = cum_sum + ((al_gmm[i] * (multivariate_normal.pdf(data.T,mean=mean_gmm[i,:], cov = cov_gmm[i,:,:]))))
    return(np.log(cum_sum).mean())

def gen_bootstrap(data,samples):
    indices = np.random.randint(0,data.shape[1],samples)
    new_data = np.zeros((2,samples))
    for i in range(samples):
        new_data[:,i] = data[:,indices[i]]
    return(new_data)

def Plot_data(X):

    #x_0=[i for i in range(0,samples) if labels[0,i]==0]
    #x_1=[i for i in range(0,samples) if labels[0,i]==1]

    plt.scatter(X[0,:],X[1,:],s=5,color='b',marker='o')
    #plt.scatter(X[0,x_1],X[1,x_1],s=2,color='b',label='class 1',marker='o')
    plt.title('GMM distribution')
    plt.xlabel('feature x1')
    plt.ylabel('feature x2')
    plt.legend()
    plt.show()

#datassets=[gen_gmm(100),gen_gmm(1000),gen_gmm(10000)]
G10,label_10=gen_gmm(10)
G100,label_100=gen_gmm(100)
G1000,label_1000=gen_gmm(1000)
G10000,label_10000=gen_gmm(10000)
#G100000=gen_gmm(100000)
#print(x)
# Plot_data(G100)
# Plot_data(G1000)
# Plot_data(G10000)
# Plot_data(G100000)
datasets=[G10,G100,G1000,G10000]
dataset_name=['G10','G100','G1000','G10000']
n_components_range = range(1, 7)
lowest_bic = np.infty
bic=[]
for dataset in datasets:
    for M in n_components_range:
        gmm = GaussianMixture(n_components=M,covariance_type='diag',max_iter=100)
        gmm.fit(dataset)
        bic.append(gmm.bic(dataset))
        # if bic[-1] < lowest_bic:
        #     lowest_bic = bic[-1]
        #     best_gmm = gmm


bic = np.array(bic).reshape(4,6)
for i in range(4):
    print(np.argmin(bic[i,:]))
label_list = ['M=1', 'M=2', 'M=3', 'M=4','M=5','M=6']
x = np.linspace(1,6,6)

rects1 = plt.bar(x=x, height=bic[0,:], width=0.1, alpha=0.8, color='k', label="G10")
rects2 = plt.bar(x=[i + 0.1 for i in x], height=bic[1,:], width=0.1, color='r', label="G100")
rects3 = plt.bar(x=[i + 0.2 for i in x], height=bic[2,:], width=0.1, color='g', label="G1000")
rects4 = plt.bar(x=[i + 0.3 for i in x], height=bic[3,:], width=0.1, color='b', label="G10000")


plt.xticks([index + 0.2 for index in x], label_list)
#plt.ylim(0, 10000)
plt.xlabel("M_components")
plt.title("BIC score")
plt.legend()
plt.show()
# spl = plt.subplot(2, 1, 1)
# for i, (dataset, color) in enumerate(zip(datasets, color_iter)):
#     xpos = np.array(n_components_range) + 0.2 * (i - 2)
#     bars.append(plt.bar(xpos, bic[i*len(n_components_range): (i + 1) * len(n_components_range)], width = .2, color = color))

# plt.xticks(n_components_range)
# plt.ylim([bic.min() * 1.01 - .01 *bic.max(), bic.max()])
# plt.title('BIC score per model')
# xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 + .2 * np.floor(bic.argmin() / len(n_components_range))
# plt.text(xpos, bic.min() * 0.97 + 0.03 * bic.max(), "*", fontsize = 14)
# spl.set_xlabel("Number of components")
# spl.legend([b[0] for b in bars], dataset_name)


# X=G100
# splot = plt.subplot(2, 1, 2)
# Y_ = clf.predict(X)

# for i, (mean, covar, color) in enumerate(zip(clf.means_, clf.covariances_, color_iter)):
#     print(covar)
#     v, w = linalg.eigh(covar)
#     if not np.any(Y_ == i):
#         continue
#     plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color = color)
    
#     angle = np.arctan2(w[0][1], w[0][0])
#     angle = 180 * angle / np.pi 
#     v *= 4
#     ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color = color)
#     ell.set_clip_box(splot.bbox)
#     ell.set_alpha(.5)
#     splot.add_artist(ell)


# plt.xlim(-10, 10)
# plt.ylim(-3, 6)
# plt.xticks(())
# plt.yticks(())
# plt.title("Selected GMM: full model, 2 components")
# plt.subplots_adjust(hspace = .35, bottom = .02)
# plt.show()

kfold= KFold(n_splits=5,random_state =None)

avg=[]
for dataset in datasets:
    for M in n_components_range:
        for train_index,test_index in kfold.split(dataset):
            temp=[]
            gmm = GaussianMixture(n_components=M,covariance_type='diag',max_iter=100)
            gmm.fit(dataset[train_index])
            temp.append(gmm.score(dataset[test_index]))
        avg.append(np.mean(temp))


avg=np.array(avg).reshape(4,6)
for i in range(4):
    print(np.argmax(avg[i,:]))
rects1 = plt.bar(x=x, height=avg[0,:], width=0.1, alpha=0.8, color='k', label="G10")
rects2 = plt.bar(x=[i + 0.1 for i in x], height=avg[1,:], width=0.1, color='r', label="G100")
rects3 = plt.bar(x=[i + 0.2 for i in x], height=avg[2,:], width=0.1, color='g', label="G1000")
rects4 = plt.bar(x=[i + 0.3 for i in x], height=avg[3,:], width=0.1, color='b', label="G10000")


plt.xticks([index + 0.2 for index in x], label_list)
plt.ylim(-100, 0)
plt.xlabel("M_components")
plt.title("average of log-likelihood")
plt.legend()
plt.show()
