% to find the accuracy and best k value for knn classifiation 

function accuracy=knn(train_data, test_data, k);
% train data contains boths the feature and the output(in the last column)
X_train=train_data(:,1:(size(train_data,2)-1));
Y_train=train_data(:,size(train_data,2));
X_test=test_data(:,1:(size(test_data,2)-1));
Y_test=test_data(:,size(test_data,2));
testcases=size(Y_test,1);
%disp(X_test);
strain=size(X_train);
for i=1:testcases
    
d=X_train-ones(strain)*diag(X_test(i));
d=abs(d(:,1));
dist=sum(d,2);
dist=[d Y_train];
sorted=sortrows(dist,1);
class(i,1)=mode(sorted(1:k,2));

end
disp(sorted);
disp(class);


c=0;
for h=1:size(class)
    if class(h)==Y_test(h,1);
    c=c+1;
    end
end 
accuracy=c*100/size(Y_test,1);
end

