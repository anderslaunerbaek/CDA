# Module 1, Exercise 4

# ex. 4 - solve diabetes using knn
library('FNN') # load FNN library for get.knnx function
library('lars') # load lars library (includes diabetes data)
data(diabetes) # load diabetes data
# str(diabetes$x) # list structure of x
# str(diabetes$y) # list structure of y
n = dim(diabetes$x)[1] # number of observations
#x = cbind(matrix(1,n,1),diabetes$x) # add a column of ones to x
p = dim(x)[2] # number of paramter estimates


K = 5; # number of neighbors in KNN

yhat = matrix(0,n,1) # initialize predicted y
X = scale(diabetes$x) # you may need to normalize variables
for (i in 1:n){ # for each observation
  KNN <- get.knnx(X,t(as.matrix(X[i,])), k=K+1, algorithm=c("kd_tree", "cover_tree", "CR", "brute")) # find the K nearest neighbors
  W = KNN$nn.dist[-1]/sum(KNN$nn.dist[-1]) # weight of each of K nearest neihgbors
  yhat[i] = W%*%diabetes$y[KNN$nn.index[-1]]; # prediction - weighted average of K nearest neighbors
}
MSE = mean((diabetes$y-yhat)^2) # Mean squared error (MSE) for the KNN model
print(MSE)
# plot the predictions as a function of the true values
plot(diabetes$y, yhat, type='p', xlab='y', ylab='yhat') 
title('KNN for diabetes')
