# Ex. 1 - ridge regression on diabetes data - model selesction strategies
rm(list=ls())
 #load data----
  library('pracma') # load pracma library for 'logspace'
  library('lars') # load lars library (includes diabetes data)
  library('cvTools') # load cvTools library (includes cvFolds)
  library('psych') # load library psych (includes tr)
  require(latex2exp)# load library latex2exp (includes TeX)
  setwd("C:/Users/federico/Google Drive/DTU/2017 Spring/Computational_Data_Analysis/Lecture2/S2")
  load("diabetes.Rdata") # load diabetes data
  n = dim(diabetes$x)[1] # number of observations
  x = as.matrix(diabetes$x) # set x as matrix
  y=(diabetes$y)
  p = dim(x)[2] # number of paramter estimates
  
#EX 1a----
  m = 100; # try m values of lambda
  lambdas = logspace(-4,2,100); # define k values of lambda on a log scale
  betas_r = matrix(0,p,m); # prepare a matrix for all the ridge parameters
  for (i in 1:m){
  betas_r[,i] = solve(t(x)%*%x+lambdas[i]*diag(p) , t(x)%*%y);
  }
  matplot(lambdas,t(betas_r),log="x",type="l",lty=1,col=1:p,ylab=TeX('$\\beta$'),xlab=TeX('$\\lambda$')
          ,main="Regularized estimates")
#EX 1b----
  m = 100; # try m values of lambda
  K = 10; # K-fold cross-validation
  x = cbind(matrix(1,n,1),x) # add a column of ones to x
  p = dim(x)[2] # number of paramter estimates
  lambdas = logspace(-4,2,100); # define k values of lambda on a log scale
  betas_r = matrix(0,p,m); # prepare a matrix for all the ridge parameters
  mse = matrix(0,m,K); # prepare vector for all the mse values
  folds <- cvFolds(n, K = K, R = 1, type = "random") # get the random folds for the CV
  for (j in 1:K){ # K fold cross-validation
    x_tr = x[folds$subsets[folds$which!=j],] # training data
    x_tst = x[folds$subsets[folds$which==j],] # test data
    y_tr = diabetes$y[folds$subsets[folds$which!=j]] # training data
    y_tst = diabetes$y[folds$subsets[folds$which==j]] # test data
    for (i in 1:m){ # repeat m times
      betas_r[,i] = solve(t(x_tr)%*%x_tr + lambdas[i]*diag(p),t(x_tr)%*%y_tr) # estimte ridge coefficients based on training data
      mse[i,j] = mean((y_tst-x_tst%*%as.matrix(betas_r[,i]))^2) # estimate mean squared error based on test data
    }
  }
  # meanMSE = apply(mse,1,mean) # find the mean mse over the K cv folds
  meanMSE=rowMeans(mse)
  Imin = which.min(meanMSE) # find the optimal value
  print(paste('Optimal CV lambda value:',lambdas[Imin]))
  # plot the optimal lambda
  lines(c(lambdas[Imin],lambdas[Imin]),c(-800,800),lwd=1.5,lty=2,col="red")
  

#EX 1c----
  # %%
  #   % Exercise 1 c %%%%%%%%%%%%%%%%
  seMSE = apply(mse,1,std)/sqrt(K);
  J = which(meanMSE[Imin] + seMSE[Imin] > meanMSE);
  j = J[length(J)];
  Lambda_CV_1StdErrRule = lambdas[j]; 
  print(paste('CV lambda 1-std-rule = ',Lambda_CV_1StdErrRule));
#EX 1d----
  # AIC and BIC (ex. 1.c) ***
  AIC = matrix(NaN,m,1)
  BIC = matrix(NaN,m,1)
  D = matrix(NaN,m,1)
  N = length(diabetes$y)    
  Beta =  solve(t(x)%*%x, t(x)%*%diabetes$y)  # OLS solution
  e = diabetes$y-x%*%Beta # the errors (residuals) of the OLS model
  s = std(e) # Low bias std of errors (estimated noise)
  
  for (j in 1:m){
    BetaR = solve(t(x)%*%x + lambdas[j]*diag(p),t(x)%*%diabetes$y)
    d = tr(x%*%solve(t(x)%*%x + lambdas[j]*diag(p))%*%t(x))
    D[j] = d # the estiamted dimensions of the model
    e = diabetes$y-x%*%BetaR # the errors of the model
    err = sum(e^2)/N # the mse of the model
    AIC[j] = err + 2*d /N * s^2 # the formula for AIC for ridge
    BIC[j] = N / s^2 * (err + log(N)* d / N * s^2) # the formula for BIC for ridge
  }
  j_aic = which.min(AIC)
  j_bic = which.min(BIC)
                           
  print(paste('AIC lambda: ',lambdas[j_aic])) # print the optimal lambda based on AIC
  print(paste('BIC lambda: ',lambdas[j_bic])) # print the optimal lambda based on BIC
  
  # plot the optimal lambdas
  lines(c(lambdas[j_aic],lambdas[j_aic]),c(-800,800),col='green') # AIC
  text(lambdas[j_aic], y = 600, labels = "AIC")
  lines(c(lambdas[j_bic],lambdas[j_bic]),c(-800,800), col='blue') # BIC
  text(lambdas[j_bic], y = 600, labels = "BIC")
#EX 1e----
# *** Bootstrap estimate of variance of parameter estimates (ex 1.d) ***
  Nboot = 100
  BetaB = array(NaN,c(p,m,Nboot))
  for (i in 1:Nboot){
  I = sample(1:n, size=n, replace = TRUE) # sample N samples from 1:N with replacement
  Xboot = x[I,]
  Yboot = diabetes$y[I]   
  for (j in 1:m){
  BetaB[,j,i] = solve(t(Xboot)%*%Xboot + lambdas[j]*diag(p),t(Xboot)%*%Yboot)
  }
  }
  BetaVar = apply(BetaB,c(1,2),var)
  
  # plot the results (log x-axis, linear y-axis), exclude the intercept
  matplot(lambdas, t(BetaVar[-1,]), type='l', log="xy", ylim=c(1e-4,1e6),lty=1,col=1:p,ylab=TeX('$\\beta$'),xlab=TeX('$\\lambda$')) 
  title('Bootstrapped variance in Ridge Regression')
  
