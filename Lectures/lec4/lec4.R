rm(list=ls())
# Linear DCA
dat <- iris
library(dplyr)

n <- dim(dat)[1]
p <- dim(dat)[2]-1

Y <- dat %>% select(Species) %>% as.matrix()
X <- dat %>% select(-Species) %>% as.matrix()

# calculate the plug-in estimate 
cat_lev <- levels(dat$Species)
k <- length(cat_lev)
#
phi_hat <- as.matrix(table(dat$Species) / n, ncol=k) 
#
mean_hat <- as.matrix(dat %>% 
    group_by(Species) %>% 
    summarise_all(funs(mean)) %>%  
    select(-Species), ncol=p) 


# X[Y==cat_lev[1],]-mean_hat[1,] %*% t(X[Y==cat_lev[1],]-mean_hat[1,])


X1 <- sweep(X[Y==cat_lev[1],], 2, mean_hat[1,])
X2 <- sweep(X[Y==cat_lev[2],], 2, mean_hat[2,])
X3 <- sweep(X[Y==cat_lev[3],], 2, mean_hat[3,])
# 
S <- matrix((t(X1) %*% X1 + t(X2) %*% X2 + t(X3) %*% X3) / (n-k), ncol = p)

y_est <- matrix(c(sweep(X %*% solve(S) %*% mean_hat[1,], 2, 0.5 * t(mean_hat[1,]) %*% solve(S) %*% mean_hat[1,] + log(phi_hat[1])),
                  sweep(X %*% solve(S) %*% mean_hat[2,], 2, 0.5 * t(mean_hat[2,]) %*% solve(S) %*% mean_hat[2,] + log(phi_hat[2])),
                  sweep(X %*% solve(S) %*% mean_hat[3,], 2, 0.5 * t(mean_hat[3,]) %*% solve(S) %*% mean_hat[3,] + log(phi_hat[3]))), 
                ncol=k)

y_hat <- cat_lev[apply(y_est,1,which.max)]
table(y_hat, Y)
