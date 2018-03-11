# Ex. 2

# *** leave-one-out cross validation ***
load('silhouettes.RData') # load silhouettes data X and class
library('cvTools') # load cvTools library (includes cvFolds)
library('class') # load library class (includes knn)
n = dim(X)[1] # number of observations
p = dim(X)[2] # number of paramter estimates

#### Plot silhouettes ####
plot(X[1,1:65],X[1,66:p],type='l', ylim = c(-0.21, 0.21), xlim= c(-0.21, 0.21),col=2, xlab = "x", ylab = "y")
for (i in 2:n){
lines(X[i,1:65],X[i,66:p], col=2+i)
}
title('Silhouettes')

#### Perform leave-one-out cross-validation ####
K = n; # leave-one-out cross-validation
k = seq(from = 1, to = 15, by=1); # define an interval for numbers of KNN
m = length(k); # try m values of k in knn
miscl = matrix(0,m,K); # prepare vector for all the misclassifications
folds <- cvFolds(n, K = K, R = 1, type = "random") # get the random folds for the CV
normalize = TRUE # normalize variables

for (j in 1:K){ # K fold cross-validation
  x_tr = X[folds$subsets[folds$which!=j],] # training data input
  x_tst = t(as.matrix(X[folds$subsets[folds$which==j],])) # test data input
  if (normalize){
    x_tr <- scale(x_tr)
    x_tst <- scale(x_tst,center = attr(x_tr, "scaled:center"), scale = attr(x_tr, "scaled:scale"))
  }
  c_tr = class[folds$subsets[folds$which!=j]] # training data output
  c_tst = class[folds$subsets[folds$which==j]] # test data output
for (i in 1:m){ # repeat m times for each value of number of neighbours
  cl_tst <- knn(train = x_tr, test = x_tst, cl = c_tr, k = k[i]) # perform knn
  miscl[i,j] = sum(cl_tst!=c_tst)/length(cl_tst) # misclassificaiton error
}
}
meanMiscl = apply(miscl,1,mean) # find the mean mse over the K cv folds
Imin = which.min(meanMiscl) # find the optimal value
print('Optimal CV k value:')
print(k[Imin])
# plot the results (log x-axis, linear y-axis), exclude the intercept
plot(k, meanMiscl, type='l',xlab="k",ylab="misclassification rate") # plot mean misclasifications for test

# plot the optimal KNN
lines(c(k[Imin],k[Imin]), c(0,1), lty = 2,col='red')
title('knn classification')