  rm(list=ls())
  dev.off()
  library(R.matlab)
  source("drawshape.R")
  library(manipulate)
# 1a load data#########################################################################################################
  
  faces=readMat(file.path("../Data/faces.mat"))
  X=faces$X
  conlist=faces$conlist

# get the size of the data
  n = dim(X)[1]
  p = dim(X)[2]
  
# 1b compute mean shape,plot and center data###########################################################################

# get the mean face
  mu = colMeans(X);
 
# plot the mean face
 drawshape(shape=data.matrix(mu), conlist=conlist,col="blue",lwd=2)
 title("Mean Face")
#center the data
  Xc = sweep(X,2,mu)
  
# 1c compute PCA,using EVD and SVD####################################################################################
 
# compute PCA as an eigenvalue analysis of the covariance matrix
  eig = eigen(cov(X), symmetric=T)
  eval=eig$values
  evec=eig$vectors

# keep only modes correspoding to strictly positive eigenvalues
  eval = eval[(eval > 1e-9)]
  evec = evec[,1:length(eval)]

# compute PCA as an SVD of the centered data matrix
  svdlist=svd(Xc, nu = min(n, p), nv = min(n, p))
  
# keep only modes correspoding to strictly positive singular values
 d = svdlist$d[svdlist$d> 1e-9]
 U = svdlist$u[,1:length(d)]
 V = svdlist$v[,1:length(d)]
 
#check if they are the same. Eigenvectors have arbitrary sign, therefore,square all their values
 print(paste("Difference in eigenvalues (variances) is", norm(eval - d^2/n,"2")))
 print(paste('Difference in eigenvectors is', norm(evec^2 - V^2,"2")))
 
# make a screeplot of the Percent Variance Explained for each singular vector
 require(svdvis)
 svd.scree(svdlist, axis.title.x = "Singular Vectors", axis.title.y = "Percent Variance Explained")
 
# 1d/e/f Plot the faces variation.################################################################################

#  assign PCA variables
#  V are the loadings
  S = U%*%diag(d); # the scores
  sigma2 = d^2/n; # the variances
drawshape(mu,conlist,T,lwd=2)
drawshape(mu + 2.5*sqrt(sigma2[1])%*%V[,1], conlist,F,col="red")
drawshape(mu - 2.5*sqrt(sigma2[1])%*%V[,1], conlist,F,col="blue")
require(latex2exp)#latex style text in plots
legend("topleft", c(TeX('$\\mu$'),TeX('$\\mu + 2.5\\sigma$'),TeX('$\\mu - 2.5\\sigma$')),lty=c(1,1,1),lwd=c(2,1,1),col=c("black","red","blue"))
# 1g    Explore the first few modes of variation using 
shape_inspector(mu,V,sigma2,lwd=2)
