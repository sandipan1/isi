function [out_hidden]=elm(input_data,hid_neu);

% load data
X=input_data;
s=size(X);

flag=0;
hidden_neuron=0;

if hid_neu <=s(1)
    hidden_neuron=hid_neu;
    flag=1;

else 
    disp('no of hidden neurons cant be more than input neurons');
    flag=0;
end
disp(X);
if flag==1
    
    
% randomize weights 
X=[ones(s(1),1) X];
input_neuron=size(X,2);
w=randi(100,hidden_neuron,input_neuron);
    %disp(size(w))
    %disp(size(X'))
    %disp(w*X');
    
H=sigmoid(w*X');        %H=h*n
beta=pinv(H')*X;        % beta=h*f
out_hidden=X*pinv(beta); % out_hidden=n*h

end


end




