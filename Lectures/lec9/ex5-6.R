rm(list=ls())
dev.off()
# install.packages("randomForest")
# install.packages("ababag")
# install.packages("xgboost")
# setwd("~/CDA02582/Lecture 9/Exercise")
require(randomForest)
require(adabag)
require(xgboost)
library(R.matlab)
library(rpart)
require(gbm)

#########################################################################################################
set.seed(433)
Z=readMat("~/DTU/Courses/CDA/Lectures/lec9/zipdata.mat")
X=Z$X
y=Z$y
N = dim(X)[1]
p = dim(X)[2]

# draw random samples
Xsample=as.matrix(X[sample(nrow(X),size=9,replace=F),])
rotate <- function(x) t(apply(x, 2, rev))
par(mfrow=c(3,3)) 
for(i in 1:9){
  M=matrix(as.numeric(Xsample[i,]),sqrt(p),byrow=T)
  w=apply(X, 1, function(x, want) isTRUE(all.equal(x, want)), Xsample[i,])
  image(rotate(M),col=grey.colors(255),main=paste("Class:",y[w]))}
##EX4 RAndom Forest #####################################################################################

# Compute random forest solution
B = 100; # The number of trees in the forest
ZIP=as.data.frame(Z)
rf=randomForest(y~.,data=ZIP,ntree=B,importance=T,proximity=T)
Xn=as.matrix(as.numeric(X))
##EX5 Bagging #####################################################################################
# parameters
CV=2 #number of cv folds
minb=5 #minimum number of leaf in a note
CP=0.01 #complexity parameter where to prune tree
ctrl=rpart.control(cp=CP,minbucket = minb)
ZIP$y=as.factor(ZIP$y)
bag=bagging.cv(y ~ .,data=ZIP, v = CV,control=ctrl,mfinal = B)

##EX6 Boosting #####################################################################################
# parameters

CV=5 #number of cv folds
maxd=6 #maximum depth of the tree
eta=.05 #learning rate
adaboost=boosting.cv(y ~ .,data=ZIP, v = CV, boos = TRUE, mfinal = B, 
                     coeflearn = "Zhu", ctrl, par=T)
boost6 <- xgb.cv(nfold=CV,prediction=T,data=xgb.DMatrix(model.matrix(y~.,data=ZIP),label=y),                   # verbose=0,
                         nrounds=B,num_class=10,
                         params=list(objective="multi:softmax",eval_metric="merror",eta=eta,max_dept=maxd))
boost4 <- xgb.cv(nfold=CV,prediction=T,data=xgb.DMatrix(model.matrix(y~.,data=ZIP),label=y),                   # verbose=0,
                 nrounds=B,num_class=10,
                 params=list(objective="multi:softmax",eval_metric="merror",eta=eta,max_dept=4))

par(mfrow=c(1,1))
bost6err=boost6$evaluation_log$test_merror_mean
bost6sdev=boost6$evaluation_log$test_merror_std
bost4err=boost4$evaluation_log$test_merror_mean
bost4sdev=boost4$evaluation_log$test_merror_std
xx <- c(B:1, 1:B)

# lines(1:B,boost$evaluation_log$test_merror_mean+boost$evaluation_log$test_merror_std)
yy6 <- c(rev(bost6err-bost6sdev), bost6err+bost6sdev)
yy4 <- c(rev(bost4err-bost4sdev), bost4err+bost4sdev)
plot   (xx, yy6, type = "n", xlab="N of trees",ylab="CV_error")
lines(seq(B),rep(adaboost$error,B),col="green")
lines(seq(B),rep(bag$error,B))
polygon(xx, yy6, col=rgb(1, 0, 0,0.2), border=NA)

polygon(xx, yy4, col=rgb(0, 0, 1,0.2), border=NA)
lines(1:B,bost6err,type="l",col="red",lwd=2)
lines(1:B,bost4err,type="l",col="blue",lwd=2)
legend("topright", legend = c(paste('XGBM_dept',maxd) ,paste('XGBM_dept',maxd,'std dev'),
                              paste('XGBM_dept',4) ,paste('XGBM_dept',4,'std dev'),"adaboost error","bagging error"),bty = "n",col = c("red",NA,"blue",NA,"green","black"),
       lty = c(1,NA,1,NA,1,1),lwd=c(2,NA,2,NA,1,1),density=c(0,NA,0,NA,0,0),fill = c("red",rgb(1, 0, 0,0.2),"blue",rgb(0, 0, 1,0.2),"green","black"),
       border = c(NA,"gray",NA,"gray",NA,NA),x.intersp=c(2,0.5,2,0.5,2,2))

boost <- xgb.train(data=xgb.DMatrix(model.matrix(y~.,data=ZIP),label=y),                   # verbose=0,
                 nrounds=B,num_class=10,
                 params=list(objective="multi:softmax",eval_metric="merror",eta=eta,max_dept=maxd))

importance=xgb.importance(feature_names = as.character(c(1:p)), model = boost )
barplot(importance$Gain[order(importance$Feature)],main="Importance of features boosting")
barplot(importance$Gain[1:20],main="most important features boosting",horiz=TRUE,names.arg=importance$Feature[1:20],las=1)
