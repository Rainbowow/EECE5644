import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import normalize
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_validate,cross_val_score
def IMG_PREPROCESS(path):
    img = cv2.imread(path)
    row=img.shape[0]
    col=img.shape[1]
    #print(img.shape)
    pixels=[]
    mmin=[sys.maxsize,sys.maxsize,sys.maxsize,sys.maxsize,sys.maxsize]
    mmax=[0,0,0,0,0]
    for i in range(img.shape[0]):#321
        for j in range(img.shape[1]):#481
            tmp=[i,j,img[i,j,0],img[i,j,1],img[i,j,2]]
            for k in range(5): #get data range
                if tmp[k]>mmax[k]:
                    mmax[k]=tmp[k]
                if tmp[k]<mmin[k]:
                    mmin[k]=tmp[k]
            pixels.append(tmp)
    
    pixels=np.array(pixels)
    mmin=np.array(mmin)
    mmax=np.array(mmax)
    mrange=mmax-mmin
    pixels=pixels/mrange
    #print(pixels)
    return pixels,row,col

img,row,col=IMG_PREPROCESS(path="3096_color.jpg")
img1,row1,col1=IMG_PREPROCESS(path="42049_color.jpg")

def GMM(pixels,row,col):
    gmm=GaussianMixture(n_components=3,max_iter=1000).fit(pixels)
    labels = gmm.predict(pixels)
    labels=labels.reshape((row,col))
    plt.imshow(labels)
    plt.show()

#GMM(img,row,col)
GMM(img1,row1,col1)

def GRIDSEARCH(components,pixels,row,col):
    best_score=-100
    best_c=0
    for i in range(2,components):
        gmm=GaussianMixture(n_components=i,max_iter=1000).fit(pixels)
        labels=gmm.predict(pixels)
        #print(labels)
        score=cross_val_score(gmm,X=pixels,y=labels,cv=10).mean()
        if score>best_score:
            best_score=score
            best_c=i
    print(best_c)
    new_gmm=GaussianMixture(n_components=best_c,max_iter=1000).fit(pixels)
    res = new_gmm.predict(pixels)
    res=res.reshape((row,col))
    plt.imshow(res)
    plt.show()

#GRIDSEARCH(5,img1,row1,col1)