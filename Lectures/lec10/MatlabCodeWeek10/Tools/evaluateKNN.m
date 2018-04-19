function errorRate=evaluateKNN(Xtrain,ytrain,Xtest,ytest,k)

% KNN classifcation
%
% Input:
%   k   number of nearest neighbours to use
%   Xtrain   N x M data matrix
%   ytrain   1 x M vector of class labels
%   Xtest    N x P data matrix
%   ytest    1 x P vector of class labels
%
% Output:
%   errorRate   The pct. of missclassified labels based on KNN

% Use KNN to classify data based on labels y
if size(Xtrain,2)*size(Xtest,2)<10^8
    d=sum(Xtest.^2)'*ones(1,size(Xtrain,2))-2*Xtest'*Xtrain+ones(size(Xtest,2),1)*sum(Xtrain.^2); % Calculate distance from training data to test data   
    [val,ind]=sort(d,2,'ascend');
    est_class=mode(ytrain(ind(:,1:k)),2);   
else
    Xtest_sq=sum(Xtest.^2);
    Xtrain_sq=sum(Xtrain.^2);
    est_class=zeros(1,size(Xtest,2));
    for t=1:size(Xtest,2)
        d=Xtest_sq(t)-2*Xtest(:,t)'*Xtrain+Xtrain_sq; % Calculate distance from training data to test data   
        [val,ind]=sort(d,2,'ascend');
        est_class(t)=mode(ytrain(ind(1:k)),2);       
    end
end

% Evaluate Error Rate
errorRate=nnz(est_class-ytest)/length(ytest);

