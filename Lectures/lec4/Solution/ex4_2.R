##############################################
##############Exercise 2b#####################
##############################################

library(glmnet)

#Load the data
GXtrain=read.csv('GolubGXtrain.csv')
GXtest=read.csv('GolubGXtest.csv')

#split data into test and train
Xtrain = GXtrain[,c(2:length(GXtrain))]
Ytrain = GXtrain[[1]] 
Xtest = GXtest[,c(2:length(GXtrain))]
Ytest = GXtest[[1]]

#Train the model and plot the result
model1=cv.glmnet(as.matrix(Xtrain), as.numeric(Ytrain),family = "binomial",nfolds = 5,standardize = T,intercept = T,alpha=1)
plot(model1)
title('Cross-validated Deviance of Lasso fit')



##############################################
##############Exercise 2c#####################
##############################################

#Finds for wich lambda we receive the smallest error 
lmin=which.min(model1$lambda)
(count0=model1$nzero[lmin])

#Finds for wich lambda we receive 1 standard deviation from the smallest error
lmin1se=which(model1$lambda==model1$lambda.1se)
(count01se=model1$nzero[lmin1se])

##############################################
##############Exercise 2d#####################
##############################################

#Evalute the accuracy
yhat=predict(model1, newx = as.matrix(Xtest), s = "lambda.1se", type = "class")
(accuracy = sum(yhat==Ytest)/length(Ytest))

