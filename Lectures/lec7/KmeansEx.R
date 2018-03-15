###################################################
################ KmeansEx.######################
rm(list=ls())
##################################################
class <- as.factor(as.matrix(read.csv("~/DTU/Courses/CDA/Lectures/lec7/data/ziplabel.csv")))
X <- read.csv("~/DTU/Courses/CDA/Lectures/lec7/data/zipdata.csv")
Kvec <- c(1:20) # Number of clusters to analyze 
W <- c()
Nsim <- 20
Wsim <- matrix(0, nrow = length(Kvec), ncol = Nsim)
N <- dim(X)[1]
p <- dim(X)[2]
minX <- apply(X, 2, min)
maxX <- apply(X, 2, max)
for (k in Kvec){
  #perform K-means
  message(paste("No. of K:",k))
  fit <- kmeans(X, k, nstart = 20)
  # get cluster means 
  meanC <- aggregate(X, by = list(fit$cluster), FUN = mean)
  #Compute within-class dissimilarity
  predC <- fit$cluster
  W <- c(W, fit$tot.withinss)
  
  #Perform Gap-statistic
  #20 simulations of data uniformly distibuted over [X]
  for (j in 1:Nsim){
    Rnumber <- matrix(runif(N * p, 0, 1), nrow = N, ncol = p)
    part3 <- (matrix(1, nrow = N, ncol = 1) %*% maxX) - (matrix(1, nrow = N, ncol = 1) %*% minX)
    Xu <- data.frame(matrix(1, nrow = N, ncol = 1) %*% minX + Rnumber * part3)
    fitSim <- kmeans(Xu, k, nstart = 20)
    Wsim[k, j] <- fitSim$tot.withinss
  }
}
#standard errors for the error bars
Elog_Wsim <- rowMeans(log(Wsim)) #expected cluster scatter
#sk=stderr(log(Wu))

par(mfrow = c(1, 2))
#plot the log within class scatters
plot(Kvec, log(W), type = "o", xlab = "number of clusters", ylab = "log W_k")
lines(Kvec, Elog_Wsim, col = 'red')
title("Within-class dissimilarity")
legend(1, 95, legend = c("Observed", "expected for simulation"))

#Plot the Gap curve
Gk  <- Elog_Wsim - log(W)
plot(Kvec, Gk, type = "o", xlab = "number of clusters", ylab = "Gk")
title("Gap curve")


# optimal 
min(which(Gk > 0)) + 1
# c(min(which(Gk > 0)), min(which(Gk > 0)) + 1)
# c(min(which(Gk > 0)), max(which(Gk <= 0)))

