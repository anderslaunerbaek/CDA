function [W,H,L,L1]=SparseCoding(X,noc,lambda,nonneg)
% Sparse Coding
%
% Written by Morten Mørup for 02582
%
% Minimize 0.5*||X-WH||_F^2+lambda*|H|_1
%
% Input
%   X       I x J data matrix
%   noc     number of components
%   lambda  L1 regularization strength
%   nonneg  constraints imposed on W and H
%           [0 0 ] -> W unconstrained, H unconstrained (default)
%           [0 1 ] -> W unconstrained, H non-negative 
%           [1 0 ] -> W non-negative, H unconstrained 
%           [1 1 ] -> W non-negative, H non-negatve 
%
% Output
%   W    I x noc non-negative matrix
%   H    noc x J non-negative matrix
%   L    Least square error of fit, i.e. 0.5*||X-XSH||_F^2
%   L1   lambda*|H|_1

[I,J]=size(X);

if nargin<4
    nonneg=[0 0];
end
if nargin<3
    lambda=0;
end
if nonneg(1)
    W=rand(I,noc);
else
    W=randn(I,noc);
end
eI=ones(I,1);
W=W./(eI*sqrt(sum(W.^2)+eps));
if nonneg(2)
    H=rand(noc,J);
else
    H=randn(noc,J);    
end

maxiter=100;
L=zeros(1,maxiter);
SST=sum(sum(X.*X));
mu_W=1e-2;
mu_H=1e-2;

for iter=1:maxiter
    
    % Update W
    [W,mu_W,L_W]=NI_PG(X*H',H*H',W,mu_W,nonneg(1),eI);
    
    % Update H
    [H,mu_H,L_H,L1_H]=SC_PG(W'*X,W'*W,H,lambda,mu_H,nonneg(2));    
    
    L(iter)=0.5*(SST+L_H);
    L1(iter)=L1_H;
    disp(['iteration ' num2str(iter) ' objective value ' num2str(L(iter)+L1(iter)) ' Pct. Variance Explained=' num2str(round((1-L(iter)/(0.5*SST))*1e5)/1e5) ]);
end


%---------------------------------------------
function [W,mu,cost]=NI_PG(XHt,HHt,W,mu,nonneg,eI)

maxiter=10;
costold=sum(sum((W'*W).*(HHt)))-2*sum(sum(W.*XHt));

for iter=1:maxiter
   G=W*HHt-XHt; 
   G=G-(eI*sum(G.*W)).*W;
   Wold=W;
   stop=0;
   while ~stop
      W=Wold-mu*G;
      if nonneg
        W(W<0)=0;           
      end
      W=W./(eI*sqrt(sum(W.^2)+eps));
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

%---------------------------------------------
function [H,mu,L,L1]=SC_PG(WtX,WtW,H,lambda,mu,nonneg)

maxiter=10;
costold=0.5*(sum(sum((WtW).*(H*H')))-2*sum(sum(H.*WtX)))+lambda*sum(sum(abs(H)));

for iter=1:maxiter
   G=WtW*H-WtX;    
   Hold=H;
   stop=0;
   while ~stop
      H=Hold-mu*G;
      H(abs(H)<mu*lambda)=0;
      H=H-mu*lambda*sign(H);            
      if nonneg
        H(H<0)=0;           
      end 
      L=sum(sum((WtW).*(H*H')))-2*sum(sum(H.*WtX));
      L1=lambda*sum(sum(abs(H)));
      cost=0.5*L+L1;
      dcost=costold-cost;            
      if dcost>=0
          mu=mu*1.2;
          costold=cost;
          stop=1;
      else
          mu=mu/2;
      end    
   end
end

