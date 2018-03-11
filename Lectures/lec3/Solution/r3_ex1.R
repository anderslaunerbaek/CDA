rm(list=ls())
set.seed(66)
library('lars') # load lars library 
library('cvTools') # load cvTools library (includes cvFolds)
library(R.matlab)#read matlab 
dat <- readMat(file.path('S5/sand.mat'))
X=dat$X # list structure of x
Y=dat$Y  # list structure of y
n = dim(X)[1] # number of observations
p = dim(X)[2] # number of paramter estimates
#crossvalidation
K = 10;    # try different numbers, n=leave-one-out CV, try also 5 and 10, and rerun to see 
# how the minimum changes for different splits of the data
folds <- cvFolds(n, K = K, R = 1, type = "random")
maxfold = round(n/K)
nsteps =length(Y) - maxfold
index = seq(nsteps)
residmat <- matrix(0, length(index), K)
errtrmat<- matrix(0, length(index), K)
for (i in seq(K)) {
  xtr = X[folds$subsets[folds$which!=i],] # training data
  xtst = X[folds$subsets[folds$which==i],] # test data
  ytr = Y[folds$subsets[folds$which!=i]] # training data
  ytst = Y[folds$subsets[folds$which==i]] # test data
  #center and normalize
  nm=dim(xtr)[1]
  one <- rep(1,nm)
  meanx <- drop(one %*% xtr)/n
  normx <- sqrt(drop(one %*% (xtr^2)))
  xtr <- scale(xtr, meanx, normx)# normalize training data subtracting mean and dividing by L2 norm
  mu <- mean(ytr)
  ytr <- drop(ytr - mu)# center training response
  ytst = drop(ytst - mu); # use the mean value of the training response to center the test response
  # # normalize test data with mean and L2 norm of training data
  xtst=scale(xtst,meanx,normx)#*matrix(1,dim(xtst)[2]),scale=F)#sd(xtr)*matrix(1,dim(xtst)[2]));
  
  #built the model
  LAR <- lars(xtr, ytr, trace = F, 
              type = "lar", normalize = F, intercept = F, use.Gram = F)
  #fit the model
  ytsthat <- predict(LAR, xtst,s = index)$fit
  ytrhat <- predict(LAR, xtr,s = index)$fit
  residmat[, i] <- apply((ytst - ytsthat)^2, 2, mean)
  errtrmat[, i] <- apply((ytr - ytrhat)^2, 2, mean)
  cat("\n CV Fold", i, "\n\n")
}
errtst <- apply(residmat, 1, mean)#cv error or test error
errtr <- apply(errtrmat, 1, mean)#training error
indexopt=which.min(errtst)
SE <- sqrt(apply(residmat, 1, var)/K)#standard error
J = which(errtst[indexopt] + SE[indexopt] > errtst)
indexSEopt = J[1];
#CP
#OLS solution using svd decomposition
Y_OLS = X%*%svd(X)$v %*% diag(1/svd(X)$d) %*% t(svd(X)$u) %*% Y;
s2 = sum((Y_OLS-Y)^2)/n;
LARSCP<- lars(X, Y, trace = F, 
              type = "lar", normalize = T, intercept = T, use.Gram = F)
Cp = scale(LARSCP$RSS/s2-n+2*LARSCP$df,F,T);
#plot
xlab= "Number of parameters";
plot(index, errtst, type = "l",lwd=2,col="red", ylim = range(errtst, errtst + SE,0),
     xlab = xlab, ylab = "Error",main="Model Selection with LARS")
error.bars(index, errtst + SE, errtst - SE, width = 1/length(index));
lines(index, errtr,lwd=2,col="blue")
lines(c(indexopt,indexopt),c(-1,20),col="red",lty=2)
lines(c(indexSEopt,indexSEopt),c(-1,20),col="green",lty=2)
lines(index,Cp[index],lwd=2,col="pink") # scale to put in the same plot
legend("topright", c("Test error","Training error","CV min MSE","1SE min MSE","scaled CP"),lty=c(1,1,2,2,1),lwd=c(2,2,1,1,2),col=c("red","blue","red","green","pink"))

