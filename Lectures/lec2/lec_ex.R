# Lecture 2
rm(list=ls())
library(dplyr)
library(ggplot2)
df <- read.csv("~/DTU/Courses/CDA/Lectures/lec1/DiabetesData.txt", sep = "")
# transformations
normalize <- function(x){ return((x - mean(x)) / (max(x) - min(x))) }
z_transf <- function(x){ (x - mean(x)) / sd(x) }

set.seed(1)

# Exersice 1 ----
# A) ----
n <- dim(df)[1]
X <- as.matrix(cbind(1,df$AGE, df$SEX, df$BMI, 
                     df$BP, df$S1, df$S2, df$S3, 
                     df$S4, df$S5, df$S6))
Y = as.matrix(df$Y)
# number of parameters
p = dim(X)[2]
k <- 100
# 
beta_est <- data.frame(matrix(NA, k, p))
colnames(beta_est) <- paste0("beta", 1:p)
l_r <- c(-4,2)
beta_est$lambda <- pracma::logspace(l_r[1], l_r[2], k)

for(ii in 1:k) {
    # estimate parameters
    beta_est[ii, 1:p] <- solve(t(X) %*% X + beta_est$lambda[ii] * diag(p)) %*% t(X) %*% Y
}

par_r <- beta_est %>% select(-lambda) %>% summarise(min = min(.),
                                                    max = max(.))
ggplot(beta_est) + 
    geom_point(aes(lambda, beta1, colour="beta1", alpha = 1/4)) +
    geom_point(aes(lambda, beta2, colour="beta2", alpha = 1/4)) +
    geom_point(aes(lambda, beta3, colour="beta3", alpha = 1/4)) +
    geom_point(aes(lambda, beta4, colour="beta4", alpha = 1/4)) +
    geom_point(aes(lambda, beta5, colour="beta5", alpha = 1/4)) +
    geom_point(aes(lambda, beta6, colour="beta6", alpha = 1/4)) +
    geom_point(aes(lambda, beta7, colour="beta7", alpha = 1/4)) +
    geom_point(aes(lambda, beta8, colour="beta8", alpha = 1/4)) +
    geom_point(aes(lambda, beta9, colour="beta9", alpha = 1/4)) +
    geom_point(aes(lambda, beta10, colour="beta10", alpha = 1/4)) +
    geom_point(aes(lambda, beta11, colour="beta11", alpha = 1/4)) +
    coord_trans(x="log") +
    scale_x_continuous(breaks = pracma::logspace(l_r[1], l_r[2], k/20)) +
    scale_y_continuous(breaks = seq(par_r$min,par_r$max, 15))


# B) ----

# apply 10-fold cross validation
K <- 10
# sep
val_size <- as.integer(n * 0.30)
tt_size <- n - val_size
# get train and test indencies
idx_val <- !(1:n %in% sample(x = 1:n, size = tt_size, replace = FALSE))
idx_tt <- !idx_val
X_tt <- matrix(X[idx_tt,], ncol = p)
Y_tt <- matrix(Y[idx_tt], ncol = 1)

folds <- caret::createFolds(which(idx_tt), K)


lambda <- pracma::logspace(l_r[1], l_r[2], k)
beta_est_mat <- array(rep(NA, K*k*p), c(K, k, p));  
MSE <- array(rep(NA, K*k*3), c(K, k, 3));  

# loop fold
for (kk in 1:K) {
    # get train folds
    idx_train <- !(which(idx_tt) %in% folds[[kk]])
    idx_test <- !idx_train
    
    # preprocess
    # X_train <- as.data.frame(X_tt[idx_train,]) %>%
    #     mutate_all(., funs(normalize)) %>%
    #     mutate(V1 = 0) %>%
    #     as.matrix()
    # X_test <- as.data.frame(X_tt[idx_test,]) %>%
    #     mutate_all(., funs(normalize)) %>%
    #     mutate(V1 = 0) %>%
    #     as.matrix()
    X_train <- as.data.frame(X_tt[idx_train,]) %>%
        mutate_all(., funs(z_transf)) %>%
        mutate(V1 = 0) %>%
        as.matrix()
    X_test <- as.data.frame(X_tt[idx_test,]) %>%
        mutate_all(., funs(z_transf)) %>%
        mutate(V1 = 0) %>%
        as.matrix()
    
    # X_train <- X_tt[idx_train,]
    # X_test <- X_tt[idx_test,]
    
    # estimate beta for given ridge 
    for(ii in 1:k) {
        # estimate parameters
        beta_r <- solve(t(X_train) %*% X_train + beta_est$lambda[ii] * diag(p)) %*% t(X_train) %*% Y_tt[idx_train]
        beta_est_mat[kk, ii, ] <- beta_r
        # caculate error
        res <- (Y_tt[idx_test] - X_test %*% beta_r)^2
        MSE[kk, ii, 1] <- mean(res)
        MSE[kk, ii, 2] <- sd(res)
        MSE[kk, ii, 3] <- var(res)
    }
}
# finde the average value for MSE for all folds as function of lambda
mse_means <- colMeans(MSE[,,1])
best_idx_mean <- which.min(mse_means)
print(paste("Optimal value of lambda:", beta_est$lambda[best_idx_mean]))
print(paste("MSE:",mse_means[best_idx_mean]))

# plot tr
cv_df <- data.frame(apply(beta_est_mat, c(2,3), mean))
cv_df$lambda <- lambda
par_r <- cv_df %>% select(-lambda) %>% summarise(min = min(.),
                                                    max = max(.))
tr_plot <- ggplot(cv_df) + 
    geom_point(aes(lambda, X1, colour="beta1", alpha = 1/4)) +
    geom_point(aes(lambda, X2, colour="beta2", alpha = 1/4)) +
    geom_point(aes(lambda, X3, colour="beta3", alpha = 1/4)) +
    geom_point(aes(lambda, X4, colour="beta4", alpha = 1/4)) +
    geom_point(aes(lambda, X5, colour="beta5", alpha = 1/4)) +
    geom_point(aes(lambda, X6, colour="beta6", alpha = 1/4)) +
    geom_point(aes(lambda, X7, colour="beta7", alpha = 1/4)) +
    geom_point(aes(lambda, X8, colour="beta8", alpha = 1/4)) +
    geom_point(aes(lambda, X9, colour="beta9", alpha = 1/4)) +
    geom_point(aes(lambda, X10, colour="beta10", alpha = 1/4)) +
    geom_point(aes(lambda, X11, colour="beta11", alpha = 1/4)) +
    geom_vline(xintercept = cv_df$lambda[best_idx_mean],colour="red") +
    coord_trans(x="log") +
    scale_x_continuous(breaks = pracma::logspace(l_r[1], l_r[2], k/20)) +
    scale_y_continuous(breaks = seq(par_r$min,par_r$max, 15))
tr_plot

# C) ----
# std. one error
sd_means <- colMeans(MSE[,,2]) / sqrt(K)
best_idx_sd <- which.min(sd_means) 
# update index
best_idx_sd <- max(which(mse_means[best_idx_sd] + sd_means[best_idx_sd] > mse_means))
print(paste("Optimal value of lambda:", beta_est$lambda[best_idx_sd]))


# plot
tr_plot <- tr_plot +
    geom_vline(xintercept = cv_df$lambda[best_idx_sd],colour="blue")
tr_plot
# you select at simpler model..

# D) ----

AIC <- BIC <- d <- rep(NA, k)
for(ii in 1:k) {
    # estimate parameters
    beta_r <- solve(t(X) %*% X + beta_est$lambda[ii] * diag(p)) %*% t(X) %*% Y
    
    # estimate effective parameters
    beta_r_tr <- X %*% solve(t(X) %*% X + beta_est$lambda[ii] * diag(p)) %*% t(X)
    d[ii] <- psych::tr(beta_r_tr)

    res <- Y - X %*% beta_r
    mse <- mean(res^2)
    sigma2 <- sd(res)^2
    
    # 
    AIC[ii] <- mse + 2 * d[ii] / n * sigma2
    BIC[ii] <- n / sigma2 * (mse + log(n) * d[ii] / n * sigma2)
}
best_idx_AIC <- which.min(AIC)
best_idx_BIC <- which.min(BIC)

print(paste("AIC: Optimal value of lambda:", beta_est$lambda[best_idx_AIC]))
print(paste("BIC: Optimal value of lambda:", beta_est$lambda[best_idx_BIC]))


tr_plot <- tr_plot +
    geom_vline(xintercept = cv_df$lambda[best_idx_AIC],colour="green") +
    geom_vline(xintercept = cv_df$lambda[best_idx_BIC],colour="orange")
tr_plot

# E) ----

beta_est_stats <- array(rep(NA, p*k*3), c(3, k, p))

bootstrap <- function(x, iter=100) {
    bs <- sample(x, size = length(x) * iter, replace = T)
    return(c(mean(bs),var(bs),sd(bs)))
}

for (kk in 1:k ){
    for (pp in 1:p ){
        beta_est_stats[, kk, pp] <- bootstrap(beta_est_mat[, kk, pp])
    }    
}

beta_est_stats_var <- data.frame(beta_est_stats[2,,])
beta_est_stats_var$lambda <- lambda

par_r <- beta_est_stats_var %>% 
    select(-lambda) %>% 
    summarise(min = min(.), max = max(.))
ggplot(beta_est_stats_var) + 
    geom_point(aes(lambda, X1, colour="beta1", alpha = 1/4)) +
    geom_point(aes(lambda, X2, colour="beta2", alpha = 1/4)) +
    geom_point(aes(lambda, X3, colour="beta3", alpha = 1/4)) +
    geom_point(aes(lambda, X4, colour="beta4", alpha = 1/4)) +
    geom_point(aes(lambda, X5, colour="beta5", alpha = 1/4)) +
    geom_point(aes(lambda, X6, colour="beta6", alpha = 1/4)) +
    geom_point(aes(lambda, X7, colour="beta7", alpha = 1/4)) +
    geom_point(aes(lambda, X8, colour="beta8", alpha = 1/4)) +
    geom_point(aes(lambda, X9, colour="beta9", alpha = 1/4)) +
    geom_point(aes(lambda, X10, colour="beta10", alpha = 1/4)) +
    geom_point(aes(lambda, X11, colour="beta11", alpha = 1/4)) +
    geom_vline(xintercept = cv_df$lambda[best_idx_mean],colour="red") +
    geom_vline(xintercept = cv_df$lambda[best_idx_sd],colour="blue") +
    geom_vline(xintercept = cv_df$lambda[best_idx_AIC],colour="green") +
    geom_vline(xintercept = cv_df$lambda[best_idx_BIC],colour="orange") +
    coord_trans(x="log") +
    scale_x_continuous(breaks = pracma::logspace(l_r[1], l_r[2], k/30)) +
    scale_y_continuous(breaks = seq(par_r$min,par_r$max, 10))

# Exersice 2 ----
rm(list = ls())
load("~/DTU/Courses/CDA/Lectures/lec2/silhouettes.RData")
Y <- factor(class)
p <- dim(X)[2]
n <- dim(X)[1]

# A) ----
# plot 
plot(X[1,1:65], X[1,66:p], type='l', 
     ylim = c(-0.21, 0.21), 
     xlim= c(-0.21, 0.21), col=2, 
     xlab = "x", ylab = "y")
for (i in 2:n){
    lines(X[i,1:65],X[i,66:p], col=2+i)
}


# select K range
K_r <- 2:20
# create cv folds 
K <- n
folds <- caret::createFolds(1:n, K)
class_rate <- array(rep(NA, K*length(K_r)*1), c(K, length(K_r), 1))
# loop fold
for (kk in 1:K) {
    # get train folds
    idx_train <- !(1:n %in% folds[[kk]])
    idx_test <- !idx_train
    
    #
    Y_test <- Y[idx_test]
    Y_train <- Y[idx_train]
    
    # preprocess
    X_train <- X[idx_train,]
    X_test <- X[idx_test,]
    
    
    for (kii in 1:length(K_r)) {
        
        # cross entropy loss
        y_hat <- class::knn(X_train, X_test, Y_train, K_r[kii])
        
        # correct classification rate
        class_rate[kk, kii, 1] <- sum(y_hat == Y_test) / length(Y_test)
        # pr1 <- y_hat*log(Y_test)
        # pr2 <- (1- y_hat) * log(1 - Y_test)
        # 
        # pr1[is.nan(pr1)] <- 0
        # pr2[is.nan(pr2)] <- 0
        # 
        # loss <- -sum(pr1+pr2)
    }
}

# get average for each fold
mean_rate <- apply(class_rate, 2, mean)
best_idx <- which.max(mean_rate)
print(paste("Classificatin rate:", mean_rate[best_idx]))
print(paste("best no. knn:", K_r[best_idx]))


# Exersice 3 ----
set.seed(2)
y <- sample(0:1, 1000, T)

y <- runif(1000)

set.seed(3)
y_true <- sample(0:1, 1000, T)


roc_data <- function(y, y_true, cut) {
    #
    NP <- sum(y_true == 1)
    NN <- sum(y_true == 0)
    TP <- sum(y[y_true == 1] > cut)
    TN <- sum(y[y_true == 0] < cut)
    
    sens <- TP / NP
    spec <- TN / NN
    # # create cm
    # cm <- matrix(table(y_true, y), ncol = length(unique(y_true)))
    # # extract diag
    # cm_diag <- diag(cm)
    # 
    # TP <- cm_diag
    # FP <- colSums(cm) - TP
    # FN <- rowSums(cm) - TP
    # TN <- sum(cm) - (FP + FN + TP)
    # 
    # #
    # sens <- TP / (TP + FN)
    # spec <- TN / (TN + FP)
    # 
    # 
    # sens_wa <- sum(sens * table(y_true) / length(y_true))
    # spec_wa <- sum(spec * table(y_true) / length(y_true))
    
    return(list("sens"=sens,
                "spec"=spec))
}


N <- 1000
roc <- data.frame(sens = rep(NA, N),
                  spec = rep(NA, N))
cut <- seq(0,1,length.out = N)

for (ii in 0:N) {
    tmp <- roc_data(y, y_true, cut[ii])
    roc$sens[ii] <- tmp$sens
    roc$spec[ii] <- 1- tmp$spec
}
plot(roc$sens, roc$spec, xlim = c(0,1),ylim = c(0,1))

roc_data(y, y_true, 0.95)
