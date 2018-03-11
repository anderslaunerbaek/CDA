##############################################
##############Exercise 1.b#####################
##############################################
library(e1071)
library(rpart)

#Load the data
Data1=read.csv('../Data/Synthetic2DOverlap.csv')
d=data.frame(x1=Data1[,1],x2=Data1[,2],y=as.factor(Data1[,3]))

#Try different values for the gamma parameter
KernelScale = 1   # <--- YOUR CHOICE <<<

#Fit svm
svm.model <- svm(y ~ .,data=d,kernel='radial',gamma=KernelScale) 

#plot result
plot(svm.model,data=d)
