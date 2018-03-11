# Module 1, Exercise 1

library('lars') # load lars library (includes diabetes data)
data(diabetes) # load diabetes data
# str(diabetes$x) # list structure of x
# str(diabetes$y) # list structure of y

# ex. 1.a:
# compute beta using the inverse
beta1 = solve(t(diabetes$x)%*%diabetes$x)%*%t(diabetes$x)%*%diabetes$y
# compute beta by solving a linear system of equations
beta2 = solve(t(diabetes$x)%*%diabetes$x,t(diabetes$x)%*%diabetes$y)
diff = beta1-beta2 # absolute difference
print(diff)

# ex. 1.b:
# include an intercept
n = dim(diabetes$x)[1] # number of observations/rows in x
x = cbind(matrix(1,n,1),diabetes$x) # add a column of ones to x
# now, compute beta with an intercept.
beta = solve(t(x)%*%x,t(x)%*%diabetes$y)
print(beta)

rss = sum((diabetes$y - x%*%beta)^2) # residual sum of squares.
mse = mean((diabetes$y - x%*%beta)^2) # mean squared error.
tss = sum((diabetes$y - mean(diabetes$y))^2) # total sum of squares.
r2 = (1 - rss/tss)*100 # R-squared. this is the amount (in %) of the variance in y which is explained by the model.
print(mse)
print(r2)

