clear;close all;clc;

data = load('iris.txt');
s = size(data,1);
randomArrayIndex = randperm(s);
testDataIndex = randomArrayIndex(1:s/5);
testData = data(testDataIndex,:);
trainDataIndex = randomArrayIndex(s/5+1:s);
trainData = data(trainDataIndex,:);
m_train = size(trainData);
m_test = size(testData);
tic
acc=knnclassifier(trainData,testData,3)

toc
%{
filename='mnist_train.csv';
trainData = importdata(filename);
filename2='mnist_test.csv';
testData = importdata(filename2);
m_train = size(trainData);
m_test = size(testData);
X_train=trainData(:,2:(m_train(2)));
X_train = (X_train - 0)/(255 - 0);
Y_train=trainData(:,1);
X_test=testData(:,2:(m_train(2)));
X_test = (X_test - 0)/(255 - 0);
Y_test=testData(:,1);
%}

X_train=trainData(:,1:(m_train(2)-1));
%X_train = (X_train - 0)/(255 - 0);
Y_train=trainData(:,m_train(2));
X_test=testData(:,1:(m_train(2)-1));
%X_test = (X_test - 0)/(255 - 0);
Y_test=testData(:,m_test(2));

hid1=2;
[out_3_train]=elm(X_train,hid1);
%size (out_3_train);
[out_3_test]=elm(X_test,hid1);

k=1
a=[Y_train X_train];
b=[Y_test X_test];
acc=knnclassifier(a,b,k);


