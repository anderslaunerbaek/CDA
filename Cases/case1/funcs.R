RMSE_func <- function(y, y_hat) { sqrt(mean((y-y_hat)^2)) / sqrt(mean((y-mean(y))^2)) }
MSE_func <- function(y, y_hat) { mean((y-y_hat)^2) }
R2_func <- function(y, y_hat){ 1 - sum((y - y_hat)^2) / sum((y - mean(y_hat))^2) }