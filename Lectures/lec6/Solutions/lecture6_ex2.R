##############################################
##############Exercise 2.a#####################
##############################################

library(corrplot)
library(glmnet)

source('ex1.R')
Lthresh=V
Lthresh[which(abs(V)<0.15)]=0
Sthresh=Xc%*%Lthresh
sigma2thresh=var(Sthresh)

drawshape(mu,conlist,T,lwd=2)
drawshape(mu + 2.5*sqrt(sigma2thresh[1])%*%Lthresh[,1], conlist,F,col="red")
drawshape(mu - 2.5*sqrt(sigma2thresh[1])%*%Lthresh[,1], conlist,F,col="blue")
title('Thresholding, 1st PA')
legend("topleft", c(TeX('$\\mu$'),TeX('$\\mu + 2.5\\sigma$'),TeX('$\\mu - 2.5\\sigma$')),lty=c(1,1,1),lwd=c(2,1,1),col=c("black","red","blue"))

require(corrplot)
par(mfrow=c(1,2))
corrplot(abs(cor(Sthresh)), method="circle")
title('Correlation of Scores')
corrplot(abs(cor(Lthresh)), method="circle")
title('Correlation of loadings')
par(mfrow=c(1,1))

##############################################
##############Exercise 2.b#####################
##############################################

v_stan = as.data.frame(scale(V))
varm=varimax(V[,1:12])
L_varimax=varm$loadings
S_varimax=Xc%*%L_varimax
sigma2_varimax=var(S_varimax)

drawshape(mu,conlist,T,lwd=2)
drawshape(mu + 2.5*sqrt(sigma2_varimax[1])%*%L_varimax[,1], conlist,F,col="red")
drawshape(mu - 2.5*sqrt(sigma2_varimax[1])%*%L_varimax[,1], conlist,F,col="blue")
title("Varimax 1st PA")
legend("topleft", c(TeX('$\\mu$'),TeX('$\\mu + 2.5\\sigma$'),TeX('$\\mu - 2.5\\sigma$')),lty=c(1,1,1),lwd=c(2,1,1),col=c("black","red","blue"))

require(corrplot)
par(mfrow=c(1,2))
corrplot(abs(cor(S_varimax)), method="circle")
title('Correlation of Scores')
corrplot(abs(cor(L_varimax)), method="circle")
title('Correlation of loadings')
par(mfrow=c(1,1))
##############################################
##############Exercise 2.c#####################
##############################################

k=12
L_en=matrix(0, nrow=p, ncol=k)
for (i in c(1:k)){
  li_en=cv.glmnet(as.matrix(Xc), as.matrix(S[,i]),nfolds = 5,standardize = T,intercept = T,alpha=0)
  L_en[,i]=coef(li_en)[c(2:(p+1))]
}

S_en=Xc%*%L_en
sigma2_en=var(S_en)

drawshape(mu,conlist,T,lwd=2)
drawshape(mu + 2.5*sqrt(sigma2_en[1])%*%L_en[,1], conlist,F,col="red")
drawshape(mu - 2.5*sqrt(sigma2_en[1])%*%L_en[,1], conlist,F,col="blue")
title('Elastic net, 1st PA')
legend("topleft", c(TeX('$\\mu$'),TeX('$\\mu + 2.5\\sigma$'),TeX('$\\mu - 2.5\\sigma$')),lty=c(1,1,1),lwd=c(2,1,1),col=c("black","red","blue"))

require(corrplot)
par(mfrow=c(1,2))
corrplot(abs(cor(S_en)), method="circle")
title('Correlation of Scores')
corrplot(abs(cor(L_en)), method="circle")
title('Correlation of loadings')
par(mfrow=c(1,1))


