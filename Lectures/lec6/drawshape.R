drawshape=function (shape, conlist,firstplot,lty,col,lwd){
# draw a shape
#shape is defined as c(x_1, ... ,x_p, y_1, ... ,y_p). 
# conlist defines the sub-parts of the shape. It is a k-by-3 matrix where each line
  # specifies c(start_index,end_index,open/closed,(0/1)). The row c(5,10,1)
  # therefore defines a closed subshape starting at index 5 and ending at index 10. 
# firstplot is an optional a logical value that syay whether or not the plot is the firstone
  # is useful in case of multiple plot default is TRUE
# lty,col, lwd are optional parameters  defines the linestyle and color of the shape default is black,1,1, e.g. 'red',1,1 for
  # a red shape drawn using a solid line of width 1.

#optional parameters
  if(missing(firstplot))
    firstplot=T
  if(missing(col))
    col="black"
  if(missing(lty))
    lty=1
  if(missing(lwd))
    lwd=1
  nPoints = length(shape)/2

  for (subShapeId in 1:dim(conlist)[1]){
    addplot=ifelse(subShapeId > 1,T,F)
    range = conlist[subShapeId, 1]:conlist[subShapeId, 2]
    subShape = cbind(shape[range],shape[range + nPoints])
    if(conlist[subShapeId, 3] == 0){
      if(!addplot & firstplot){
        plot(subShape[, 1], subShape[, 2],lwd=lwd, col=col,lty=lty,type="l",
             ylim=c(-0.25,0.25),xlim=c(-0.25,0.25),
             xlab="", ylab="")
      }else{
        lines(subShape[, 1], subShape[, 2],col=col,lty=lty,lwd=lwd)}
    }else{
      closedShape = rbind(subShape, subShape[1,]);
      if(!addplot){
        plot(closedShape[, 1], closedShape[, 2], col=col,lwd=lwd,type="l",lty=lty,ylim=c(-0.25,0.25),xlim=c(-0.25,0.25))
      }else{
        lines(closedShape[, 1], closedShape[, 2], col=col,lwd=lwd,lty=lty)}
    }
  }
}

shape_inspector=function (mu,V,sigma2,lty,col,lwd){
  # mu is the mean face
  # V is the right eigen vectors (loadings) matrix obtainen by SVD
  # sigma2 is the vector of variances of principal component
  # firstplot is an optional a logical value that syay whether or not the plot is the firstone
  # is useful in case of multiple plot default is TRUE
  # lty,col, lwd are optional parameters  defines the linestyle and color of the shape default is black,1,1, e.g. 'red',1,1 for
  # a red shape drawn using a solid line of width 1.
  #optional parameters
  if(missing(col))
    col="black"
  if(missing(lty))
    lty=1
  if(missing(lwd))
    lwd=1
  max1=5
  min1=-5  
  manipulate::manipulate(drawshape(mu+PCA1*sqrt(sigma2[1])%*%V[,1]+PCA2*sqrt(sigma2[2])%*%V[,2]+PCA3*sqrt(sigma2[3])%*%V[,3]
                       +PCA4*sqrt(sigma2[4])%*%V[,4]+PCA5*sqrt(sigma2[5])%*%V[,5]+PCA6*sqrt(sigma2[6])%*%V[,6]
                       , conlist,lwd=lwd,col=col,lty=lty),  
             PCA1=manipulate::slider(min1, max1, step = 0.5,initial = 0),
             PCA2=manipulate::slider(min1, max1, step = 0.5,initial = 0),
             PCA3=manipulate::slider(min1, max1, step = 0.5,initial = 0),
             PCA4=manipulate::slider(min1, max1, step = 0.5,initial = 0),
             PCA5=manipulate::slider(min1, max1, step = 0.5,initial = 0),
             PCA6=manipulate::slider(min1, max1, step = 0.5,initial = 0)
  )
}