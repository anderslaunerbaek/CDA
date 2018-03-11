rm(list=ls())
# Load data

Tb = read.csv('../Data/Ex2Data.csv',header=F);
colnames(Tb)=c("X1","X2","Y")
X=data.matrix(Tb[,1:2])
Tb[,3]=as.factor(Tb[,3])
Y=Tb[,3]
#
# Parameters to the Support Vector Machine
#
KernelFunction  = "radial";   #c("linear","polynomial","radial","sigmoid")
KernelScale     = 3;      # <--- YOUR CHOICE parameter needed for all kernels except linear
PolynomialOrder = 6;       # YOUR CHOICE, used when 'polynomial'
BoxConstraint   = 20;      #cost of constraints violation (default: 1)
                            # -it is the 'C'-constant of the regularization term in the Lagrange formulation.

svm.model=svm(Y~.,data = Tb,type="C-classification", kernel=KernelFunction,gamma=KernelScale,cost=BoxConstraint,scale = TRUE
      ,degree=PolynomialOrder)
plot(svm.model, Tb)


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
contour(x1,x2,matrix(ZZ,length(x1),length(x2)),level=1,lty=2,lwd=1.5,drawlabels=F,add=T)#supportvectors boundary
contour(x1,x2,matrix(ZZ,length(x1),length(x2)),level=-1,lty=2,lwd=1.5,drawlabels=F,add=T)#supportvectors boundary
