rm(list=ls())
# Load data

Tb = read.csv('../Data/ACS.csv');
Tb$Gender=as.factor(Tb$Gender)
Itrain = Tb$Train==1;
Train=Tb[Itrain,-dim(Tb)[2]]
Test=Tb[!Itrain,-dim(Tb)[2]]
Y_train = Tb$Y[Itrain];
Y_test  = Tb$Y[!Itrain];
X_train = Tb[Itrain,1:(dim(Tb)[2]-2)];
X_test  = Tb[!Itrain,1:(dim(Tb)[2]-2)];
Y=as.factor(Tb$Y)
X=as.matrix(X_train)

#Logistic regression
B=glm(Y~.,family = binomial,data =Train)
  # perfect separation Warning message:
  # glm.fit: fitted probabilities numerically 0 or 1 occurred 

#elastic net(ridge) logistic regression is more accurate doesn´t have problems with perfect separation
#increasing the value of alpha make the function slower but can reduce the number of paramiters
library(glmnet)
B1=cv.glmnet(data.matrix(X_train), as.numeric(Y_train),family = "binomial",alpha=0,nfolds=5)

yhat=predict(B, newdata = Test,type = "response")>0.5
accuracylogreg = sum(yhat==Y_test)/length(Y_test)
yhat1=predict(B1, newx = data.matrix(X_test), type = "response")>0.5
accuracyellogreg = sum(yhat1==Y_test)/length(Y_test)
print(paste('Accuracy by logistic regression =',round(accuracylogreg,4)))
print(paste('Accuracy by penalized logistic regression =',round(accuracyellogreg,4)))

#  Build a svm model.
#  Use the svm function library(e1071)
library(e1071)
