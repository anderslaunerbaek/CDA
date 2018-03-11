rm(list = ls())
setwd("~/DTU/Courses/CDA/Lectures/lec6/")
source("drawshape.R")
tmp <- R.matlab::readMat("faces.mat")
dat <- tmp$X
conlist <- tmp$conlist
rm(tmp)

p <- dim(dat)[2] / 2
n <- dim(dat)[1]

# 2 ----
# Compute the mean shape and center the data
mean_shape <- colMeans(dat)
dat_cen <- dat - mean_shape
# Plot the mean shape using
drawshape(shape = mean_shape, conlist = conlist,col="blue",lwd=2)


# 3 ----
# Compute a principal component analysis of the data. 
eps <- 1e-9
# EVD 
eig <- eigen(cov(as.matrix(dat)), symmetric = T)
# keep only modes correspoding to strictly positive eigenvalues
eval=eig$values[(eval > eps)]
evec=eig$vectors[,(eval > eps)]


# svd
s <- svd(as.matrix(dat_cen), nu = min(n, p), nv = min(n, p))
# keep only modes correspoding to strictly positive singular values
d <- s$d[s$d > eps]
U <- s$u[, s$d > eps]
V <- s$v[, s$d > eps]

svdvis::svd.scree(s, axis.title.x = "Singular Vectors", axis.title.y = "Percent Variance Explained")

# Try using both an eigen value decomposition (EVD) and a singular value decomposition (SVD). 
# Remember that the EVD is computed on the correlation or covariance matrix and the SVD on the data matrix itself.



#  assign PCA variables
#  V are the loadings
S <- U %*% diag(d) # the scores
sigma2 <- d^2/n # the variances
drawshape(shape = mean_shape, conlist = conlist,lwd=2)
drawshape(mean_shape + 2.5 * sqrt(sigma2[1]) %*% V[,1], conlist,F,col="red")
drawshape(mean_shape - 2.5 * sqrt(sigma2[1]) %*% V[,1], conlist,F,col="blue")

shape_inspector(mean_shape, V, sigma2, lwd=2)

