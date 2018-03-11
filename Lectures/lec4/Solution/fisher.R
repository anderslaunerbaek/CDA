T=read.csv("../Data/FisherIris.csv")
X = as.matrix(T[,1:4]);
Y = as.factor(T[,5])
N = dim(X)[1];
Yclass=levels(Y);

pi1=sum(Y==Yclass[1])/N
pi2=sum(Y==Yclass[2])/N
pi3=sum(Y==Yclass[3])/N


mu1 = colMeans(X[Y==Yclass[1],])
mu2 = colMeans(X[Y==Yclass[2],])
mu3 = colMeans(X[Y==Yclass[3],])

x1=sweep(X[Y==Yclass[1],],2,mu1) #subtract vector from matrix
x2=sweep(X[Y==Yclass[2],],2,mu2)
x3=sweep(X[Y==Yclass[3],],2,mu3)


S = (t(x1)%*%x1+t(x2)%*%x2+t(x3)%*%x3)/(N-3);

 
d1 = sweep(X%*%solve(S)%*%mu1,2,(1/2)*mu1%*%solve(S)%*%mu1-log(pi1)); #subtracting log(pi1) inside sweep in like adding
d2 = sweep(X%*%solve(S)%*%mu2,2,(1/2)*mu2%*%solve(S)%*%mu2-log(pi2));
d3 = sweep(X%*%solve(S)%*%mu3,2,(1/2)*mu3%*%solve(S)%*%mu3-log(pi3));

Yhat = apply(cbind(d1,d2,d3),1,which.max)
C = matrix(0,3,3)
for (i in 1:N)
   C[Y[i],Yhat[i]] = C[Y[i],Yhat[i]]  + 1;  

print('Confusion matrix')
print(C)
colnames(C)=Yclass
rownames(C)=Yclass
require(reshape2)
melted_C <- melt(C[,3:1])
library(ggplot2)
ggplot(data = melted_C, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile(color = "white")+geom_text(aes(Var2, Var1, label = value), color = "black", size = 4)+
  scale_fill_gradient2(low = "white", high = "purple", mid = "green",midpoint = 25, limit = c(0,max(C))      )+
  # theme(axis.title.x =element_blank(),axis.title.y =element_blank())
scale_y_discrete(name="Actual Class") + scale_x_discrete(name="Predicted Class")