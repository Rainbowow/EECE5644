import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset, DataLoader
from skorch import NeuralNetClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def GENERATE_DATA(samples,mean,cov):
    #labels
    #4 classes with uniform priors 
    labels=np.random.randint(0,3+1,samples)

    #data
    X=np.zeros((samples,3))
    for i in range(0,samples):
        if(labels[i]==0):
            X[i,:]=np.random.multivariate_normal(mean[0],cov,1)
        elif(labels[i]==1):
            X[i,:]=np.random.multivariate_normal(mean[1],cov,1)
        elif(labels[i]==2):
            X[i,:]=np.random.multivariate_normal(mean[2],cov,1)
        else:
            X[i,:]=np.random.multivariate_normal(mean[3],cov,1)

    return X,labels

def PLOT_DATA(X,samples,labels):
    ax=plt.axes(projection='3d')

    x_0=[i for i in range(0,samples) if labels[i]==0]
    x_1=[i for i in range(0,samples) if labels[i]==1]
    x_2=[i for i in range(0,samples) if labels[i]==2]
    x_3=[i for i in range(0,samples) if labels[i]==3]

    ax.scatter3D(X[x_0,0],X[x_0,1],X[x_0,2],s=5,color='r',label='class 0',marker='*')
    ax.scatter3D(X[x_1,0],X[x_1,1],X[x_1,2],s=2,color='b',label='class 1',marker='o')
    ax.scatter3D(X[x_2,0],X[x_2,1],X[x_2,2],s=5,color='g',label='class 2',marker='*')
    ax.scatter3D(X[x_3,0],X[x_3,1],X[x_3,2],s=2,color='grey',label='class 2',marker='o')
    plt.title('Data distribution')
    ax.set_xlabel('feature x1')
    ax.set_ylabel('feature x2')
    ax.set_zlabel('feature x3')
    plt.legend()
    plt.show()

#data distribution
mean=np.zeros((4,3))
#cov=np.zeros(4,3,3)
for i in range(0,4):
    mean[i,:]=[4*i,3,3]
cov=np.zeros((3,3))
for i in range(0,3):
    cov[i,i]=1

#generate and plot data
D_train_100,train_labels_100=GENERATE_DATA(100,mean,cov)
D_train_200,train_labels_200=GENERATE_DATA(200,mean,cov)
D_train_500,train_labels_500=GENERATE_DATA(500,mean,cov)
D_train_1000,train_labels_1000=GENERATE_DATA(1000,mean,cov)
D_train_2000,train_labels_2000=GENERATE_DATA(2000,mean,cov)
D_train_5000,train_labels_5000=GENERATE_DATA(5000,mean,cov)
D_test_10000,test_labels_10000=GENERATE_DATA(10000,mean,cov)
train_datasets=[(D_train_100,train_labels_100),(D_train_200,train_labels_200),(D_train_500,train_labels_500),(D_train_1000,train_labels_1000),(D_train_2000,train_labels_2000),(D_train_5000,train_labels_5000)]
# PLOT_DATA(D_train_100,100,train_labels_100)
# PLOT_DATA(D_train_200,200,train_labels_200)
# PLOT_DATA(D_train_500,500,train_labels_500)
# PLOT_DATA(D_train_1000,1000,train_labels_1000)
# PLOT_DATA(D_train_2000,2000,train_labels_2000)
# PLOT_DATA(D_train_5000,5000,train_labels_5000)
# PLOT_DATA(D_test_10000,10000,test_labels_10000)#shape (10000,3)

# Train_data=torch.from_numpy(D_train_1000).type(torch.FloatTensor)
# Target_data=torch.from_numpy(train_labels_1000).type(torch.LongTensor)
#MLP
class MLP(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(MLP, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.out(x)
        x=F.softmax(x,dim=1)
        return x

def train(model,data_x,target): 
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
    loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted

    plt.ion()
    for t in range(200):  # loop over the dataset multiple times
        out=model(data_x)
        loss=loss_func(out,target)
        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
        if t % 2 == 0:
            # plot and show learning process
            ax=plt.axes(projection='3d')
            plt.cla()
            prediction = torch.max(out, 1)[1]
            pred_y = prediction.data.numpy()
            #print(pred_y)
            target_y = target.data.numpy()
            #print(target_y)
            ax.scatter3D(data_x.data.numpy()[:, 0], data_x.data.numpy()[:, 1],data_x.data.numpy()[:, 2], c=pred_y, s=100, lw=0, cmap='RdYlGn')
            accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
            print(accuracy)
            #plt.text(1.5, -4,s= 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
            plt.pause(0.1)

    print('Finished Training')
    plt.ioff()
    plt.show()

def TRAIN_AND_VALIDATE(model,data_x,target):
    net = NeuralNetClassifier(
    model,
    max_epochs=100,
    lr=0.1,
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,
    )
    #model selection
    net.fit(data_x,target)
    np.savetxt("data.txt",cross_val_score(net, data_x.astype(np.float32), target.astype(np.int64), cv=10,scoring='accuracy'),delimiter=',') 
    
def TRAIN_AND_VALIDATE1(data_x,target,P):
    net=MLPClassifier(hidden_layer_sizes=(P,4),activation='relu',solver='adam',learning_rate_init=0.1,max_iter=1000)
    train_result=cross_val_score(net, data_x.astype(np.float32), target.astype(np.int64),cv=10,scoring='neg_log_loss')
    #print(result)
    #pred_result=net.predict(D_test_10000)
    return train_result


    #np.savetxt("data.txt",cross_val_score(net, data_x.astype(np.float32), target.astype(np.int64), cv=10,scoring='accuracy'),delimiter=',')
#train_and_validate(MLP(n_feature=3, n_hidden=100, n_output=4),D_train_1000.astype(np.float32), train_labels_1000.astype(np.int64))
#TRAIN_AND_VALIDATE1(D_train_100,train_labels_100)

def GRIDSEARCH():
    res=[]

    for X,y in train_datasets:
        for P in range(1,16):
            print('perceptron=',P,'data size=',y.shape)
            # print(X,y)
            train_loss=TRAIN_AND_VALIDATE1(X,y,P)
            res.append(np.mean(train_loss))
            
        #draw loss picture
        X_label=np.arange(1,16)
        plt.bar(X_label,res,facecolor = '#9999ff',edgecolor = 'white')
        
        for x,r in zip(X_label,res):
            #ha : horizontal alignment
            #va : vertical alignment
            plt.text(x + 0.01,r+0.01,'%.2f'%r,ha = 'center',va='top')

        #plt.xlim(-.5,10)
        plt.xticks(X_label)
        #plt.ylim(-2,0)
        #plt.yticks([])
        plt.xlabel('perceptrons')
        plt.ylabel('train log_loss')
        plt.title('train log_loss in datasize %d'%y.shape[0])
        plt.savefig('train_log_loss in datasize %d.png'%y.shape[0])
        plt.cla()
        #plt.show()
        res.clear()

#GRIDSEARCH()
def ASSESS():
    pred_acc=[]
    for P,dataset in zip([7,13, 9, 10, 2, 8],train_datasets):
        #print(P,dataset[0])
        net=MLPClassifier(hidden_layer_sizes=(P,4),activation='relu',solver='adam',learning_rate_init=0.1,max_iter=1000).fit(dataset[0],dataset[1])
        pred_res=net.predict(D_test_10000)
        pred_acc.append(accuracy_score(test_labels_10000,pred_res))
    # draw accuracy picture
    print(pred_acc)
    X_label=np.arange(1,7)
    plt.cla()
    plt.bar(X_label,pred_acc,facecolor = '#9999ff',edgecolor = 'white')
    
    for x,r in zip(X_label,pred_acc):
        #ha : horizontal alignment
        #va : vertical alignment
        plt.text(x + 0.01,r+0.01,'%.2f'%r,ha = 'center',va='top')

    #plt.xlim(-.5,10)
    plt.xticks(X_label,['D100','D200','D500','D1000','D2000','D5000'])
    #plt.ylim(-2,0)
    plt.yticks([])
    plt.xlabel('dataset')
    plt.ylabel('prediction accuracy')
    plt.title('prediction accuracy in all dataset')
    #plt.show()
    plt.savefig('prediction_accuracy_in_all_dataset.png')
    plt.cla()
ASSESS()