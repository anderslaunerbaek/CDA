rm(list=ls())
library(rpart)
library(rpart.plot)
library(party)

# Read data, see ClevelandHeartData.txt for details

tab= read.csv2('../Data/ClevelandHeartData.csv')
# y=as.factor(tab[,14])
assign(colnames(tab)[14],as.factor(tab[,14]))
X= tab[,-14]
X[,c(2,3,6,7,9,11,13)]=lapply(X[,c(2,3,6,7,9,11,13)], factor)
Xlabel   = colnames(X)
Nobs     = nrow(X)
Nfeature = ncol(X)
#Build and view a large tree
fit = rpart(Diagnosis~.,data=tab,method = "class",control =rpart.control(minbucket=1, cp=0,xval=10))
prp(fit,varlen=3)


printcp(fit) # display the results 
plotcp(fit) # visualize cross-validation results 
# select the compleity parameter cp that minimimize the cv error withb 1 se
CVerr=fit$cptable[,"xerror"]
minCP=fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"]
minSE=min(CVerr)+fit$cptable[which.min(CVerr),"xstd"]
w=which.min(CVerr[CVerr>minSE])
wv=which(CVerr>minSE)[w]
CP_SE=fit$cptable[wv,"CP"]
#Prune the tree:
pruned.tree <- prune(fit, cp = CP_SE)
prp(pruned.tree,varlen=3)
printcp(pruned.tree) # display the results 

# Accuracy of the pruned tree as 1-the cross-validated error rate
rooterror=sum(Diagnosis==1)/Nobs
Accuracy= 1-rooterror*fit$cptable[wv,"xerror"]
cat("\n Accuracy",Accuracy,"\n")

# confusion matrix 
conf.matrix <- table(Diagnosis, predict(pruned.tree,type="class"))
rownames(conf.matrix) <- paste("Actual", rownames(conf.matrix), sep = ":")
colnames(conf.matrix) <- paste("Pred", colnames(conf.matrix), sep = ":")
print(conf.matrix)
