import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
import random as ran

pd.set_option('display.max_columns',5)
pd.set_option('display.max_rows',200)
data=pd.read_csv("iris.csv")

matrix=[[i for i in range(3)] for j in range(150)]
plt.scatter(data["sepal.length"],data["sepal.width"],data["petal.length"])
# plt.show()
X=data.iloc[:,[0,1,2,3]].values


m = X.shape[0]
n = X.shape[1]
K=3
Centroids = np.array([]).reshape(4, 0)
for i in range(K):
    rand = ran.randint(0, m - 1)
    Centroids = np.c_[Centroids, X[rand]]


EuclidianDistance=np.array([]).reshape(m,0)

for k in range(3):
       tempDist=np.sum((X-Centroids[:,k])**2,axis=1)


       EuclidianDistance=np.c_[EuclidianDistance,tempDist]
       C=np.argmin(EuclidianDistance,axis=1)+1


for i in range(1000):

    EuclidianDistance = np.array([]).reshape(m, 0)
    for k in range(K):
        tempDist = np.sum((X - Centroids[:, k]) ** 2, axis=1)
        EuclidianDistance = np.c_[EuclidianDistance, tempDist]
    C = np.argmin(EuclidianDistance, axis=1) + 1
    # step 2.b
    Y = {}
    for k in range(K):
        Y[k + 1] = np.array([]).reshape(4, 0)
    for i in range(m):
        Y[C[i]] = np.c_[Y[C[i]], X[i]]

    for k in range(K):
        Y[k + 1] = Y[k + 1].T

    for k in range(K):
        Centroids[:, k] = np.mean(Y[k + 1], axis=0)


color=['red','blue','green','cyan','magenta']
labels=['cluster1','cluster2','cluster3']

for k in range(K):
    plt.scatter(Y[k+1][:,0],Y[k+1][:,1],c=color[k],label=labels[k])
plt.scatter(Centroids[0,:],Centroids[1,:],s=300,c='black',label='Centroids')

plt.legend()
plt.show()