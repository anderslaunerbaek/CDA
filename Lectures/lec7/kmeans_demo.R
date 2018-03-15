###################################################
################K.MEANS_DEMO######################
##################################################
#SIMULATE DATA
rm(list = ls())
library(dplyr)
library(ggplot2)
#dimensions
p <- 2
#number of datapoints
N <- 500
#Number of mixture components
K <- 10

#Simulate data from a gaussian mixture distributions
muX1 <- 7 * runif(K, 0, 1) #pick any random number, the higher it is the more
muX2 <- 7 * runif(K, 0, 1) #seperated the data will be. Here I tried 7
sds <- sqrt(rep(1, K))
df_plot <- data.frame(components = as.factor(sample(1:K, prob = rep(1 / K, K), size = N, replace = TRUE)))
df_plot$samplesX1 <- rnorm(n = N, mean = muX1[df_plot$components], sd = sds[df_plot$components])
df_plot$samplesX2 <- rnorm(n = N, mean = muX2[df_plot$components], sd = sds[df_plot$components])

ggplot(df_plot) +
    geom_point(aes(samplesX1, samplesX2, colour = components))
##################################################################
#K.MEANS APPLIED ON THE SIMULATED DATA
X <- cbind(df_plot$samplesX1, df_plot$samplesX2)
fit <- kmeans(X, centers = K, nstart = 20)
# get clusters means
meanC <- aggregate(X, by = list(fit$cluster), FUN = mean) %>% 
    mutate(Group.1 = as.factor(Group.1))
# cunfusion table
table(fit$cluster, df_plot$components)
#plot the cluster means onto the photo
ggplot(df_plot) +
    geom_point(data = df_plot, aes(samplesX1, samplesX2, colour = as.factor(fit$cluster))) +
    geom_point(data = meanC, aes(V1, V2, fill = Group.1), size = 4, shape = 23) +
    labs(fill = "Centroids", shape = "", colour = "Group")
