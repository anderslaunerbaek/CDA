##############################################
##############Exercise 1.a#####################
##############################################

library(e1071)
library(rpart)

#Load the data
Tb=read.csv('../Data/Synthetic2DNoOverlapp.csv')
colnames(Tb)=c("X1","X2","Y")
X= as.matrix(Tb[,c(1:2)])
Tb$Y=as.factor(Tb$Y)
Y=Tb$Y
#FIT SVM
ker="radial"#try different types of kernels. The options are: 'polynomial', 'linear', 'radial' or 'sigmoid' 
deg=3 #degree: parameter needed if the kernel is polynomial (default: 3)
gamm=3 #gamma: parameter needed for all types of kernels except linear (default: 1/(data dimension));
svm.model <- svm(Y ~ .,data=Tb,degree=deg,kernel=ker,gamma=gamm) 

#plot result
plot(svm.model,data=Tb)#default plot not very clear

#create mesh grid
n=50
  grange = apply(X, 2, range)
  x1 = seq(from = grange[1, 1], to = grange[2, 1], length = n)
  x2 = seq(from = grange[1, 2], to = grange[2, 2], length = n)
  grid=expand.grid(X1 = x1, X2 = x2)
Ygridd = predict(svm.model, grid)
#find decision buondary
dec=predict(svm.model, grid,decision.values = T)
ZZ=as.vector(attributes(dec)$decision)
# Plot
plot(grid, col = c( "black","red")[as.numeric(Ygridd)], pch = 20, cex = 0.2)
points(X, col = Y , pch = 19)
points(X[svm.model$index, ], pch = 5, cex = 2)#supportvectors
contour(x1,x2,matrix(ZZ,length(x1),length(x2)),level=0,lwd=2,drawlabels=F,add=T)#decision boundary

