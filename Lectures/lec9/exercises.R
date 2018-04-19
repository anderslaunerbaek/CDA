rm(list = ls())

# install.packages("randomForest")
# install.packages("R.matlab")
library(dplyr)
# 4)


X <- R.matlab::readMat("./Lectures/lec9/zipdata.mat")$X %>% as.data.frame(.)
Y <- R.matlab::readMat("./Lectures/lec9/zipdata.mat")$y %>% as.vector(.) %>% as.factor(.)
test_size <- 0.3
n <- dim(X)[1]
p <- dim(X)[2]

set.seed(22)
idx_test <- 1:n %in% sample(1:n, size = n * test_size)
X_test <- X[idx_test,]
Y_test <- Y[idx_test]
X_train <- X[!idx_test,]
Y_train <- Y[!idx_test]

# fit model 

# Some heuristics for choosing parameters � Classification
mtry <- floor(sqrt(p)) # classification
# m = floor(p/3) # regression

rf <- randomForest::randomForest(x = X_train, y = Y_train,
                           #xtest = X_test, ytest = Y_test,
                           ntree = 500,
                           mtry=mtry,
                           importance=TRUE, 
                           proximity=TRUE)

pred <- predict(rf, X_test, type="response")


table(Y_test, pred)




