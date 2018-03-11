require(glmnet)
library('cvTools') # load cvTools library (includes cvFolds)
library(R.matlab)#read matlab 
rm(list=ls())
# set.seed(66)
dat <- readMat(file.path('sand.mat'))
require(covTest)
X=dat$X # list structure of x
Y=dat$Y  # list structure of y
n = dim(X)[1] # number of observations
p = dim(X)[2] # number of paramter estimates
l=15
alphas=seq(0,1,length.out = l)
  # alpha=1 ridge
  # alpha=0 lasso
  # 0<alpha<1 elastic net
err=matrix(0,l,1)
lambdas=matrix(0,l,3)
nonzeroparameter=matrix(0,l,1)
colnames(lambdas)=c("lambda","lambda1","lambda2")
for (i in seq(l)){
a=alphas[i];
fit=cv.glmnet(X, Y,alpha=a,  type.measure = "mse", nfolds = 5,standardize = T, 
              intercept = T)
err[i,]=min(fit$cvm)
plot.cv.glmnet(fit)
title(paste("CV MSE", " ","alpha=",round(a,2)," "))
lambdas[i,1]=fit$lambda[which.min(fit$cvm)]
lambdas[i,2]=lambdas[i,1]*a
lambdas[i,3]=lambdas[i,1]*(1-a)/2
nonzeroparameter[i]=fit$nzero[which.min(fit$cvm)]
cat("\n loop", i, " ","alpha=",a," ","lambda=",lambdas[i,1])
}
# the selection of alpha is not based on CV 
a=alphas[which.min(err[,1])]
fit=cv.glmnet(X, Y,alpha=a,  type.measure = "mse", nfolds = 5)
lambda=fit$lambda[which.min(fit$cvm)]

