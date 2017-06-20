%load dataset 
clear; close all;clc;
data = load('iris.txt');
s = size(data,1);
randomArrayIndex = randperm(s);
testDataIndex = randomArrayIndex(1:s/5);
testData = data(testDataIndex,:);
trainDataIndex = randomArrayIndex(s/5+1:s);
trainData = data(trainDataIndex,:);
m_train = size(trainData);
m_test = size(testData);
X_train=trainData(:,1:(m_train(2)-1));
Y_train=trainData(:,m_train(2));
X_test=testData(:,1:(m_train(2)-1));
Y_test=testData(:,m_test(2));

% enter the number of hidden layer and get its output

hid1=3;
out_3_train=elm(X_train,hid1);
%size (out_3_train);
out_3_test=elm(X_test,hid1);



hid2=2;
out_2_train=elm(X_train,hid2);
out_2_test=elm(X_test,hid2);



%implement KNN for 3 hidden layers

k=3;
out_3_train=[out_3_train Y_train];
out_3_test=[out_3_test Y_test];
acc=knn(out_3_train,out_3_test,k)

% kNN for 2 hidden layers 
%{
s=3
out_2_train=[out_2_train Y_train];
out_2_test=[out_2_test Y_test];
acc=knn(out_2_train,out_2_test,3)
%}











