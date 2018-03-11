# Module 1, Exercise 2

# ex. 2.a
n = 10; p = 3 # number of observations, number of variables
beta_true = as.matrix(c(1, 2, 3)) # these are the coefficients of the true model
X = runif(n*p, min = 0, max = 1) # % this is the data matrix, considered fixed here
dim(X) <- c(n,p)

m = 100 # m repetitions of our experiment
betas = matrix(0,p,m) # prepare a matrix for all the coefficient vectors (m of them)

sigma = 0.1 # noise level
for (i in 1:m){ # repeat m times
  y = X%*%beta_true + sigma*rnorm(n, mean=0, sd = 1); # measured response is the true model plus noise
  betas[,i] = solve(t(X)%*%X,t(X)%*%y); # estimate coefficients, should be close the true ones
}

# plot betas, average values should correspond to the true values
# plot using a boxplot
boxplot(t(betas))
