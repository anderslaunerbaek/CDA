function [W,H,L,L_LS]=NMFKL(X,noc,maxiter,W,H)
% Non-negative matrix factorization based on Multiplicative Updates
%
% Minimize \sum_{i,j} (WH)_ij log(x_(ij)/(WH)_ij)+x_ij-(WH)_ij  s.t. W>=0, H>=0
%
% Input
%   X    I x J data matrix
%   noc  number of components
%
% Output
%   W    I x noc non-negative matrix
%   H    noc x J non-negative matrix
%   L    Least square error of fit, i.e. 0.5*||X-XSH||_F^2



[I,J]=size(X);
if nargin<3
    maxiter=50;
end
if nargin<4
    W=rand(I,noc);
end
if nargin<5
    H=rand(noc,J);
end

L=zeros(1,maxiter);
L_LS=L;
SST=sum(sum(X.*X));
R=W*H;
for iter=1:maxiter
          
    % Update W (add eps to all the denominators to avoid dividing by zero)    
    
    
    
    % Update H (add eps to all the denominators to avoid dividing by zero)
    
    
    
    % Evaluate objective function
    R=W*H;
    L(iter)=sum(sum(X.*log((X+eps)./(R+eps))))-sum(sum(X))+sum(sum(R));
    L_LS(iter)=0.5*sum(sum((R-X).^2)); % Calculate LS error for comparison to KL
    fprintf('iteration %2.0f objective value %8.2f  pct. explained variance %3.1f \n', iter,L(iter),(1-L_LS(iter)/SST)*100);
end