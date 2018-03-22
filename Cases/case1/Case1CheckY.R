Case1CheckY=function(yhat){    
  tab=read.csv("~/DTU/Courses/CDA/Cases/case1/Case1_Answer.csv")
  y = tab$Y[101:1100]
  if(class(yhat)=="data.frame"){yhat=as.matrix(yhat)}
  if(length(yhat)!=1000 ){
  cat('\n error Yhat should have length 1000\n')
  return(NA)
  }else if(sum(is.na(yhat))>0){
    cat('\n error Yhat have missing values NaN\n')
    return(NA)
    }else{
      yhat = as.numeric(yhat) #Always a numeric vector
  rRMSE = sqrt(mean((y-yhat)^2)) / sqrt(mean((y-mean(y))^2))
  cat('\nYour prediction error has rRMSE =',rRMSE,"\n");
  return(rRMSE)
  }
}