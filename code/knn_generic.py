import numpy as np
from sklearn import datasets
import pandas as pd
import random
import csv
import matplotlib.pyplot as plt 

# read the dataset half_kernel 
	
csvf=open("/home/sandipan/deeplearning/isi/Datasets/Outlier.txt",'r')
lines=csv.reader(csvf)
lines=list(lines)
for i in range(len(lines)):
    for j in range(len(lines[i])):
        lines[i][j]=float(lines[i][j])
linearr=np.array(lines)
print linearr.shape
a=1
if a==1:
# randomize it and split it into test and train data
	data=linearr[:]
	a=[i for i in range(data.shape[0])]
	random.shuffle(a)

	train=(0.8*data.shape[0])
	train=int(train)
	# print train


	test=data.shape[0]-train
	X_train=np.array([data[a[i]][:-1] for i in range((train))])
	X_test=np.array([data[a[i]][:-1] for i in range((test))])
	Y_train=np.array([data[a[i]][-1:] for i in range((train))])
	Y_test=np.array([data[a[i]][-1:] for i in range((test))])
	#print X_train.shape,X_test.shape,Y_test.shape,Y_train.shape


	class kNN:
		def __init__(self):
			pass
		def training(self,X,Y):
			self.x_train=np.array(X)
			self.y_train=np.array(Y)
			
			
			
		def prediction(self,X,k=1):
			X=np.array(X)
			testcases=len(X)
			arr=[]
			
			for i in xrange (testcases):
				# manhattan distance or L1 distance
				# dist has list of L1 distance from all the training data for ith  test data
				dist=np.sum(np.abs(self.x_train[:,:]-X[i,:]),axis=1)
				#dist=dist.tolist()
				
				array=[]
				
				for j in range(len(dist)):
					array.append(([self.y_train[j],dist[j]]))
	#             print array
				#print new_dist
				#sorting according to distance from the ith training data
				
				sorted_dist= sorted (array,key=lambda x:x[1])
				sorted_dist=np.array(sorted_dist)
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
	if __name__=='__main__':
		l_train=len(Y_train)
		l_test=len(Y_test)
		cvfold=5
		limit=l_train/cvfold
		cross=[(i*limit,(i+1)*limit) for i in range(cvfold) ]
		k=[1,3,11,31,121,201]
		acc_kvalues=[]
		for i in k:
			acc_cross=[]
			
			for j in cross:
				new_xtrain=[]
				new_xtest= X_train[j[0]:j[1]][:]
				new_xtrain1=X_train[0:j[0]][:]
				new_xtrain2=X_train[j[1]:][:]
				
				new_xtrain=np.concatenate((new_xtrain1,new_xtrain2)) # only list operation  is possible No numpy


				
				new_ytrain=[]
				new_ytest=Y_train[j[0]:j[1]][:]
				
				new_ytrain1=Y_train[0:j[0]][:]
				new_ytrain2=Y_train[j[1]:][:]
				new_ytrain=np.concatenate((new_ytrain1,new_ytrain2))
				ob=kNN()
				ob.training(new_xtrain,new_ytrain)
				arr=ob.prediction(new_xtest,i)
				accu=ob.accuracy(new_ytest,arr)
				acc_cross.append(accu)
			#print acc_cross    
			avg_accu=np.mean(acc_cross)
				 #print acc_cross
				
			acc_kvalues.append(avg_accu)
		print acc_kvalues
		#p=max(enumerate(acc_kvalues),key=lambda x:x[1])
		#print 'Max Accuracy :'+str(p[1])+' for k='+str(k[p[0]])	