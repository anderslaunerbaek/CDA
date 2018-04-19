function [W,H,L,cpu_time]=NMFPG(X,noc,maxiter,W,H)
% Sparse Coding
%
% Written by Morten Mørup for 02582
%
% Minimize 0.5*||X-WH||_F^2
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
    maxiter=100;
end
if nargin<4
    W=rand(I,noc);
end
if nargin<5
    H=(W'*W)\(W'*X);
    H(H<0)=0;
end
L=zeros(1,maxiter);
cpu_time=L;
SST=sum(sum(X.*X));
mu_W=1;
mu_H=1;

for iter=1:maxiter
    tic;
    % Update W
    [W,mu_W,L_W]=NMF_PG(X*H',H*H',W,mu_W);
    
    % Update H
    [Ht,mu_H,L_H]=NMF_PG(X'*W,W'*W,H',mu_H);
    H=Ht';
    
    % rescale W and H
    d=sqrt(sum(W.^2));
    W=W*diag(1./(d+eps));
    H=diag(d)*H;
    
    L(iter)=0.5*(SST+L_H);
    cpu_time(iter)=toc;
    disp(['iteration ' num2str(iter) ', objective value ' num2str(L(iter)) ' Pct. Variance Explained=' num2str(round((1-L(iter)/(0.5*SST))*1e5)/1e5) ]);
end
% Normalize W
d=sqrt(sum(W.^2));
W=W*diag(1./(d+eps));
H=diag(d)*H;

%---------------------------------------------
function [W,mu,cost]=NMF_PG(XHt,HHt,W,mu)

maxiter=10;
costold=sum(sum((W'*W).*(HHt)))-2*sum(sum(W.*XHt));

for iter=1:maxiter
   G=W*HHt-XHt; 
   h=diag(HHt)';
   G=G.*h(ones(size(W,1),1),:);
   Wold=W;
   stop=0;
   while ~stop
      W=Wold-mu*G;
      W(W<0)=eps;           
      cost=sum(sum((W'*W).*(HHt)))-2*sum(sum(W.*XHt));
      dcost=costold-cost;            
      if dcost>=-1e-12*abs(cost)
          mu=mu*1.2;
          costold=cost;
          stop=1;
      else
          mu=mu/2;
      end    
   end
end
