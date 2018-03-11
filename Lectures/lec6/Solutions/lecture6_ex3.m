close all, clear
load sand

% make a pls regression with 20 components with 10-fold CV and 5 monte
% carlo repetitions
[XL,yl,XS,YS,beta,PCTVAR,MSE,stats] = plsregress(X,Y,20,'cv',10,'mcreps',5);

% plot the cross validation error
figure;
plot(0:20,MSE(2,:),'-bo');
xlabel('Number of components')
ylabel('Average MSE of 10-fold CV')

% plot the explained variance
figure;
plot(1:20,cumsum(100*PCTVAR(2,:)),'-bo');
xlabel('Number of PLS components');
ylabel('Percent Variance Explained in y');

% make a pls regression with 10 components
[XL,yl,XS,YS,beta,PCTVAR,MSE,stats] = plsregress(X,Y,10,'cv',10,'mcreps',5);

% compute fitted values and display residuals
yfit = [ones(size(X,1),1) X]*beta; % beta contains the PLS regression coefficients
residuals = Y-yfit;
MSE = sum(residuals.^2)/(length(residuals)-1)

% plot residuals
figure;
stem(residuals)
xlabel('Observation');
ylabel('Residual');

% plot the loadings - in order to compare these the variables should be
% normalized or at least on a comparable scale.
figure;
[Z, mu, sigma] = zscore(X); % this normalizes X, and we can use the standard deviation sigma to normalize vbeta
stem(beta(2:2017)./sigma') % don't plot the intercept value - the actual loadings for each component are found in stats.W
xlabel('Variable number')
ylabel('Relative imporatance in PLS model')

