function [W,S,H,L]=ArchetypalAnalysis(X,noc)

% Written by Morten Mørup for 02582
%
% Minimize 0.5*||X-XSH||_F^2 s.t. |s_d|_1=1, S>=0, |h_j|_1=1, H>=0
%
% Usage
%    [W,S,H,L]=ArchetypalAnalysis(X,noc)
% Input
%   X    I x J data matrix
%   noc  number of components
%
% Output
%   W    feature matrix given by X*S of size I x noc  
%   S    feature matrix of size J \times noc giving convex combinations of data points that form
%        features
%   H    noc\times J matrix relating each observation to the extracted
%        features XS
%   L    Least square error of fit, i.e. 0.5*||X-XSH||_F^2



[I,J]=size(X);
% Initialize by FurthestSum
i=FurthestSum(X,noc,1);
S=zeros(J,noc);
for t=1:noc
    S(i(t),t)=1;
end
H=log(rand(noc,J));
H=H./(ones(noc,1)*sum(H));

maxiter=50;
L=zeros(1,maxiter);
SST=sum(sum(X.*X));
mu_H=1e-2;
mu_S=1e-2;
W=X*S;
for iter=1:maxiter
    
    % Update H
    [H,mu_H,L_H]=PG_AA_H(W'*X,W'*W,H,mu_H);
    
    % Update S
    XHt=X*H';
    [S,W,mu_S,L_S]=PG_AA_S(X,X'*XHt,H*H',S,mu_S);
    
    L(iter)=0.5*(SST+L_S);
    disp(['iter ' num2str(iter) ' objective value ' num2str(L(iter)) ' Pct. Variance Explained=' num2str(round((1-L(iter)/(0.5*SST))*1e5)/1e5) ]);
end

if size(W,1)==2 && noc<6 % Sort components for 2D data
    ind=sortAAsolution(W);
    W=W(:,ind);
    S=S(:,ind);
    H=H(ind,:);
end


%----------------------------------------------------
function [H,mu,cost]=PG_AA_H(WtX,WtW,H,mu)

[noc,I]=size(H);
e=ones(noc,1);
maxiter=25;
costold=sum(sum((WtW).*(H*H')))-2*sum(sum(H.*WtX));

for iter=1:maxiter
   G=WtW*H-WtX;
   G=G-e*sum(G.*H);
   Hold=H;
   stop=0;
   while ~stop
      H=Hold-mu*G;
      H(H<0)=0; 
      l1_H=sum(H);
      H=H./(e*l1_H);
      cost=sum(sum((WtW).*(H*H')))-2*sum(sum(H.*WtX));
      dcost=costold-cost;
      if dcost>-1e-12*abs(cost)
          mu=mu*1.2;
          costold=cost;          
          stop=1;
      else
          mu=mu/2;
      end    
   end
end

%----------------------------------------------------
function [S,XS,mu,cost]=PG_AA_S(X,XtXHt,HHt,S,mu)

[J,noc]=size(S);
e=ones(J,1);
maxiter=25;
XS=X*S;
costold=sum(sum((XS'*XS).*(HHt)))-2*sum(sum(XtXHt.*S));

for iter=1:maxiter
   G=(X'*XS)*HHt-XtXHt;
   G=G-e*sum(G.*S);
   Sold=S;
   stop=0;
   while ~stop
      S=Sold-mu*G;
      S(S<0)=0; 
      l1_S=sum(S);
      S=S./(e*l1_S);
      XS=X*S;
      cost=sum(sum((XS'*XS).*(HHt)))-2*sum(sum(XtXHt.*S));
      dcost=costold-cost;
      if dcost>-1e-12*abs(cost)
          mu=mu*1.2;
          costold=cost;          
          stop=1;
      else
          mu=mu/2;
      end    
   end
end

%--------------------------------------------------------
function ind=sortAAsolution(W)
noc=size(W,2);
per=perms(1:noc);
K=size(per,1);
vol=zeros(K,1);
for k=1:K
    for t=1:noc
        if t<noc
            vol(k)=vol(k)+det([W(:,per(k,t)),W(:,per(k,t+1))]);
        else
            vol(k)=vol(k)+det([W(:,per(k,t)),W(:,per(k,1))]);
        end
    end    
end
[val,i]=max(vol);
ind=per(i,:);    
    
    
