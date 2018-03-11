# Module 1, Exercise 3

# ex. 3.b - solve diabetes using ridge
library('pracma') # load pracma library for 'logspace'
library('lars') # load lars library (includes diabetes data)
data(diabetes) # load diabetes data
# str(diabetes$x) # list structure of x
# str(diabetes$y) # list structure of y
n = dim(diabetes$x)[1] # number of observations
x = cbind(matrix(1,n,1),diabetes$x) # add a column of ones to x
p = dim(x)[2] # number of paramter estimates

k = 100; # try k values of lambda
lambdas = logspace(-4,3,k); # define k values of lambda on a log scale
betas_r = matrix(0,p,k); # prepare a matrix for all the ridge parameters
for (i in 1:k){ # repeat m times
  betas_r[,i] = solve(t(x)%*%x + lambdas[i]*diag(p),t(x)%*%diabetes$y) # estimte ridge coefficients
}
# plot the results (log x-axis, linear y-axis), exclude the intercept
plot(lambdas, betas_r[2,], log="x", type='l',ylim=c(-800,800)) # ignore log warning, it plots in log scale just fine
for (j in 3:p){
lines(lambdas, betas_r[j,], log="x")
}
title('Ridge Regression');


# ex. 3.c - bias and variance for ridge (simulated data)
n = 10; p = 3 # number of observations, number of variables
beta_true = as.matrix(c(1, 2, 3)) # these are the coefficients of the true model
X = runif(n*p, min = 0, max = 1) # % this is the data matrix, considered fixed here
dim(X) <- c(n,p)

m = 100 # m repetitions of our experiment
betas_ridge = matrix(0,p,m) # prepare a matrix for all the coefficient vectors (m of them)
sigma = 0.1 # noise level
lambda = 0.1 # regularization parameter for ridge
for (i in 1:m){ # repeat m times
  y = X%*%beta_true + sigma*rnorm(n, mean=0, sd = 1); # measured response is the true model plus noise
  betas_ridge[,i] = solve(t(X)%*%X + lambda*diag(p),t(X)%*%y); # estimate ridge coefficients, should be close the true ones
}

# plot betas, average values should correspond to the true values
# plot using a boxplot
boxplot(t(betas_ridge))

