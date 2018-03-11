rm(list=ls())
library(dplyr)
library(glmnet)
# load data
dat_train <- read.csv("~/DTU/Courses/CDA/Lectures/lec4/GolubGXtrain.csv", header = F)
X_train <- dat_train %>% select(-V1) %>% as.matrix(.)
Y_train <- dat_train %>% select(V1) %>% as.matrix(.)

dat_test <- read.csv("~/DTU/Courses/CDA/Lectures/lec4/GolubGXtest.csv", header = F)
X_test <- dat_test %>% select(-V1) %>% as.matrix(.)
Y_test <- dat_test %>% select(V1) %>% as.matrix(.)
rm(dat_test, dat_train)

# ex 2 A) 
# - YES 

# ex 2 
# - b 

# fit <- glmnet(X_train, Y_train, family="binomial", standardize = T, intercept = T)

fit <- cv.glmnet(X_train, Y_train, family="binomial", nfolds = 5,
                 standardize = TRUE, intercept = TRUE, alpha = 1)

# find the 1 std. error
"lambda.1se"


# print(fit)
plot(fit, xvar = "lambda", label = TRUE)
test <- coef(fit, s = "lambda.1se")
coef(fit, s = "lambda.1se")

Y_hat <- as.integer(predict(fit, newx = X_test, s = "lambda.1se", type = "class"))


cm <- matrix(table(Y_hat, Y_test), length(unique(Y_test)))
cm
accurcy <- sum(diag(cm)) / sum(cm) * 100
accurcy
