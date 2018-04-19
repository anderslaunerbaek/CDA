function [W,H,L,cpu_time]=NMFLS(X,noc,maxiter,W,H)
% Non-negative matrix factorization based on Multiplicative Updates
%
% Minimize 0.5*||X-WH||_F^2 s.t. W>=0, H>=0
%
% Input
%   X       I x J data matrix
%   noc     number of components
%   maxiter maximum number of iterations
%
% Output
%   W           I x noc non-negative matrix
%   H           noc x J non-negative matrix
%   L           Least square error of fit, i.e. 0.5*||X-XSH||_F^2
%   cpu_time    pr. iteration cost



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
cpu_time=L;
SST=sum(sum(X.*X));

HHt=H*H';
for iter=1:maxiter
    tic;      
    % Update W
    XHt=X*H';    
    for k=1:10
        W=W.*XHt./(W*HHt+eps);
    end
    
    
    % Update H
    WtX=W'*X;
    WtW=W'*W;
    for k=1:10
        H=H.*WtX./(WtW*H+eps);
    end
    
    
    % Evaluate objective function
    HHt=H*H';
    L(iter)=0.5*(SST+sum(sum((WtW).*(HHt)))-2*sum(sum(WtX.*H)));
    
    cpu_time(iter)=toc;
    fprintf('iter %2.0f objective value %8.2f  pct. explained variance %3.1f \n', iter,L(iter),(1-L(iter)/SST)*100);
end