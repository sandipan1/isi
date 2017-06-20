clear;close all;clc;
data = load('iris.txt');
s = size(data,1);
randomArrayIndex = randperm(s);
testDataIndex = randomArrayIndex(1:s/5);
testData = data(testDataIndex,:);
trainDataIndex = randomArrayIndex(s/5+1:s);
trainData = data(trainDataIndex,:);
m_train = size(trainData,1);
m_test = size(testData,1);
%We will train a SLP with 5 input(one of them is bias) and 3 output layers for 3 labels 0,1,2 respectively 
%----------------------------------------------------------------
x = trainData(:,1:4);
x1 = ones(4*s/5,1);
x = [x1';x(:,1)';x(:,2)';x(:,3)';x(:,4)']';
x = x';
    target = zeros(120,3);
for i = 1:m_train

if(trainData(i,5) == 0)
target(i,1) = 1;
else
target(i,1) = 0;
end

if(trainData(i,5) == 1)
target(i,2) = 1;
else
target(i,2) = 0;
end

if(trainData(i,5) == 2)
target(i,3) = 1;
else
target(i,3) = 0;
end

end
target = target';
%Now the inputs and targets are declared 
%------------------------------------------------------------------
%Weights are initialized----------
w = rand(5,3); %First row is for the biases,so weight = 1
%---------------------------------

%Training the SLP model-----------------------------------------------
learningRate = 0.005; %You can change it
maxIter = 1000;  %and it
output = zeros(3,120);
%for i = 1:maxIter
flag=true;
while flag
z = w'*x;
ai = sigmoid(z);
%{
for j = 1:3
for k = 1:m_train
if(ai(j,k)>=0.5)
output(j,k) = 1;
else
output(j,k) = -1;
end
end
end
%}
output = ai;
difference = target-output;

delta_w = learningRate*(difference*x');
w = w + delta_w';
maxim=max(max(abs(difference)));
if (maxim<0.0005)
    flag=false
end
end
%----------------------------------------------------------------------------
X = [ones(1,s/5);testData(:,1)';testData(:,2)';testData(:,3)';testData(:,4)']';
X = X';
actual_output = testData(:,5);
Z = w'*X;
probability = sigmoid(Z);
probability = probability';
%Prediction and Accuracy------------------------------------------------
correct = 0;
for i = 1:m_test
p0 = probability(i,1);p1 = probability(i,2);p2 = probability(i,3);
p = [p0 p1 p2];
if(max(p) == p0)
predict = 0;
end
if(max(p) == p1)
predict = 1;
end
if(max(p) == p2)
predict = 2;
end

if(actual_output(i,1)==predict)
correct = correct+1;
end
end
ACCURACY = correct/m_test
%---------------------------------------------------------------------
  