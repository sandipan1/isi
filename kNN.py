import numpy
from sklearn import datasets
import pandas as pd
import random

# load dataset 
iris=datasets.load_iris()

# randomize dataset
ar=[x for x in range(150)]
random.shuffle(ar)
#splitting test and train data
X_train =[iris.data[ar[i]]  for  i in range(120)]
Y_train =[iris.target[ar[i]]  for i in range(120)]
X_test=[iris.data[ar[i]] for i in range(120,150)]
Y_test=[iris.target[ar[i]] for i in range(120,150)]

class kNN:
    def __init__(self):
        pass
    def training(self,X,Y):
        self.x_train=numpy.array(X)
        self.y_train=numpy.array(Y)
        
        
        
    def prediction(self,X,k=1):
        X=numpy.array(X)
        testcases=len(X)
        arr=[]
        
        for i in xrange (testcases):
            # manhattan distance or L1 distance
            # dist has list of L1 distance from all the training data for ith  test data
            dist=numpy.sum(numpy.abs(self.x_train[:,:]-X[i,:]),axis=1)
            #dist=dist.tolist()
            
            array=[]
            
            for j in range(len(dist)):
                array.append(([self.y_train[j],dist[j]]))
#             print array
            #print new_dist
            #sorting according to distance from the ith training data
            
            sorted_dist= sorted (array,key=lambda x:x[1])
            sorted_dist=numpy.array(sorted_dist)
            # c contains the class values
            #print sorted_dist
            c=[]
            for m in range(k):
                c.append(sorted_dist[m][0])
#             if i==0:
#                 print sorted_dist[:,:]
                
            # finding which c value has max repetition
            classified =max(c,key=c.count)
            arr.append(classified)
        
        return arr
        
    def accuracy(self,test_data,predicted_data):
        c=0
        for i in range(len(test_data)):
            if test_data[i]==predicted_data[i]:
                c+=1
        ac = (c*100/len(test_data))
        return ac
    
cross=[(0,24),(24,48),(48,72),(72,96),(96,120)]
k=[1,3,5]
acc_kvalues=[]
for i in k:
    acc_cross=[]
    
    for j in cross:
        new_xtrain=[]
        new_xtest= X_train[j[0]:j[1]][:]
        new_xtrain1=X_train[0:j[0]][:]
        new_xtrain2=X_train[j[1]:][:]
        new_xtrain=new_xtrain1+new_xtrain2
        
        
        new_ytrain=[]
        new_ytest=Y_train[j[0]:j[1]][:]
        
        new_ytrain1=Y_train[0:j[0]][:]
        new_ytrain2=Y_train[j[1]:][:]
        new_ytrain=new_ytrain1+new_ytrain2
        
        ob=kNN()
        ob.training(new_xtrain,new_ytrain)
        arr=ob.prediction(new_xtest,i)
        accu=ob.accuracy(new_ytest,arr)
        acc_cross.append(accu)
    print acc_cross    
    avg_accu=numpy.mean(acc_cross)
    
    
    acc_kvalues.append(avg_accu)
print acc_kvalues