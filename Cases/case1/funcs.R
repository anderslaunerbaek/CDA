RMSE_func <- function(y, y_hat) { sqrt(mean((y-y_hat)^2)) / sqrt(mean((y-mean(y))^2)) }
MSE_func <- function(y, y_hat) { mean((y-y_hat)^2) }
R2_func <- function(Y_train, Y_hat){
    RSS <- sum((Y_train - Y_hat)^2)
    TSS <- sum((Y_train - mean(Y_hat))^2)
    return(1 - RSS / TSS)
}