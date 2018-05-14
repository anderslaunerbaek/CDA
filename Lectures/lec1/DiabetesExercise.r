rm(list=ls())
library(dplyr)
df = read.csv("~/DTU/Courses/CDA/Lectures/lec1/DiabetesData.txt", sep = "")
set.seed(1)

# exercise 1 ----
# A)
# the smart apporach
lm.ols <- lm(Y ~ AGE + SEX + BMI + BP + S1 + S2 + S3 + S4 + S5 + S6, data = df)
summary(lm.ols)

coef(lm.ols)

# define matrices
X <- model.matrix(lm.ols)
# X = as.matrix(cbind(1,df$AGE, df$SEX, df$BMI, 
#                     df$BP, df$S1, df$S2, df$S3, 
#                     df$S4, df$S5, df$S6))
Y = as.matrix(df$Y)

# calculate OLS
theta = solve(t(X)%*%X)%*%t(X)%*%Y

df_theta <- data.frame(theta_lm = coef(lm.ols), 
                       theta_mat = as.vector(theta)) %>% 
    mutate(d = (theta_lm - theta_mat) / theta_lm * 100)

# B)
# - add 1 in the first row of the design matrix

# C)
MSE <- mean((Y - X %*% theta)^2)
# d 
RSS <- sum((Y - X %*% theta)^2)
TSS <- sum((Y - mean(X %*% theta))^2)
R2 <- 1 - RSS / TSS 

# exercise 2 ----
p <- 3
n <- 10

theta_true <- matrix(c(1,2,3), nrow = p)
X <- matrix(runif(n*p, min = 0, max = 1), nrow = n)

no_repeat <- 100
sigma <- 0.1
# capture all estimates 
theta_est <- matrix(NA, no_repeat, p)
for(ii in 1:no_repeat) {
    # estimate y
    Y <- X %*% theta_true + sigma * rnorm(n, 0, 1)
    # OLS 
    theta_est[ii, ] <- solve(t(X)%*%X)%*%t(X)%*%Y
}
# compare with box plot
theta_est %>% boxplot(.) 
theta_est %>% summary(.)

# exercise 3 ----
# B)
X = as.matrix(cbind(1,df$AGE, df$SEX, df$BMI, 
                    df$BP, df$S1, df$S2, df$S3, 
                    df$S4, df$S5, df$S6))
Y = as.matrix(df$Y)
p = dim(X)[2]
k <- 100
pen <- seq(from=0.0001, to=1000, length.out = k)
theta_est <- matrix(NA, k, p)

for(ii in 1:k) {
    pen_ii <- pen[ii] * diag(p)
    # OLS with pen
    theta_est[ii, ] <- solve(t(X) %*% X + pen_ii) %*% t(X) %*% Y
}

# create plot
plot(pen, theta_est[,2], type = "l", ylim=c(-80,80), log = "x")
for (j in 3:p){
    lines(pen, theta_est[, j])
}

# C)
p <- 3
n <- 10

X <- matrix(runif(n*p, min = 0, max = 1), nrow = n)
sigma <- 0.01
lambda <- 0.1
# capture all estimates 
theta_est <- matrix(NA, n, p)
for(ii in 1:n) {
    # estimate y
    Y <- X %*% theta_true + sigma * rnorm(n, 0, 1)
    # OLS with pen
    theta_est[ii, ] <- solve(t(X) %*% X + lambda * diag(p)) %*% t(X) %*% Y
}
# compare with box plot
theta_est %>% boxplot(.) 
theta_est %>% summary(.)

# exercise 4 ----


normalize <- function(x){
    return((x- mean(x))/(max(x)-min(x)))
}

X <- df %>% select(-Y) %>% 
    mutate_all(funs(normalize)) %>% 
    as.matrix(.)
Y <- df %>% select(Y) %>% as.matrix(.)






KNN_reg <- function(X, Y, K = 5){
    # define number of parameters
    p = dim(X)[2]
    # define number of observations
    n = dim(X)[1]
    # initilize empty matrix
    # idx, yhat, dist, KNNs
    y_hat <- data.frame(matrix(c(1:n, rep(NA, (K + 3 ) * n)), ncol = K + 4))
    colnames(y_hat) <- c("idx", "y", "yhat", "dist", paste0("K", c(1:K)))
    y_hat$y <- as.vector(Y)
    # estimate each observation 
    for(ii in 1:n) {
        y_hat$dist <- NA
        # calculate distance
        for(jj in 1:n) { y_hat$dist[jj] <- sqrt(sum((X[ii,] - X[jj, ])^2)) }
        #
        dist_knn <- y_hat %>% filter(idx != ii) %>% select(idx, dist) %>% arrange(dist) %>% head(K)
        # insert knn
        y_hat[ii, paste0("K", c(1:K))] <- dist_knn$idx
        # get 
        y_hat$yhat[ii] <- matrix(dist_knn$dist / sum(dist_knn$dist), nrow = 1) %*% Y[dist_knn$idx]
    }
    #
    return(y_hat)
}

for (k in c(3,10, 25)) {
    tmp_df <- KNN_reg(X, Y, K=k)
    MSE = mean((tmp_df$y-tmp_df$yhat)^2) # Mean squared error (MSE) for the KNN model
    print(MSE)
    # plot the predictions as a function of the true values
    plot(tmp_df$y, tmp_df$yhat, type='p', xlab='y', ylab='yhat') 
}


