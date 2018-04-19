function [W,H,L,L1,Predmis,varExp]=SparseCodingMissingData(X,M,noc,lambda,nonneg,W,H,maxiter)
% Low rank missing data approximation
%
% Minimize 0.5*||X-WH||_F^2+lambda*|H|_1
% s.t. ||w_d||_2=1 for all d
%
% Input
%   X       I x J data matrix
%   M       I x J sparse matrix indicating missing values
%   noc     number of components
%   lambda  L1 regularization strength
%   nonneg  constraints imposed on W and H
%           [0 0 ] -> W unconstrained, H unconstrained (default)
%           [0 1 ] -> W unconstrained, H non-negative (default)
%           [1 0 ] -> W non-negative, H unconstrained (default)
%           [1 1 ] -> W non-negative, H non-negatve (default)
%
% Output
%   W       I x noc non-negative matrix
%   H       noc x J non-negative matrix
%   L       Least square error of fit, i.e. 0.5*||X-WH||_F^2
%   L1      lambda*|H|_1
%   Predmis estimated missing values
%   varExp  variance explained, i.e. (1-||X-WH||_F^2/||X||_F^2)*100
%
% Written by Morten Mørup
% This code is provided as is without any kind of warranty
[I,J]=size(X);
[Im,Jm]=find(M);
X=X-M.*X;
if nargin<5
    nonneg=[0 0];
end
if nargin<4
    lambda=0;
end
if nargin<6
    if nonneg(1)
        W=rand(I,noc);
    else
        W=randn(I,noc);
    end
end
W=bsxfun(@times,W,1./bsxfun(@max,sqrt(sum(W.^2,1)),eps));

if nargin<7
    H=(W'*W)\(W'*X);
    if nonneg(2)
        H(H<0)=0;            
    end
end

if nargin<8
    maxiter=1000;
end
L=zeros(1,maxiter);
L1=zeros(1,maxiter);
SST=sum(sum(X.*X));
mu_W=1/mean(sum(H.*H,2));
mu_H=1/mean(sum(W.*W,1));
iter=0;
cost_old=inf;
dcost=inf;
tol=1e-9;
disp('----------------------------------------------------------------------------');
disp('Iter. | Obj. Val. | dObj/ObjVal. | Expl. Var. |    W-step     |       H-step');
disp('----------------------------------------------------------------------------');
while iter<maxiter && dcost>=tol*cost_old 
    iter=iter+1;
    
    % Update W
    [W,mu_W]=NI_PG(X*H',H*H',W,H,Im,Jm,mu_W,nonneg(1),lambda);
    
    % Update H
    [H,mu_H,L_H,L1_H]=SC_PG(W'*X,W'*W,W,H,lambda,Im,Jm,mu_H,nonneg(2));    
    
    % Rescale problem if no regularization 
    if lambda==0 && sum(nonneg)==0
        [Uw,Sw,Vw]=svd(W,'econ');       
        W=Uw;
        H=Sw*Vw'*H;
    elseif lambda==0       
        scale=bsxfun(@max,sqrt(sum(W.^2,1)),eps);
        W=bsxfun(@times,W,1./scale);
        H=bsxfun(@times,H,scale');
    end
    
    
    % display result of iteration
    L(iter)=0.5*(SST+L_H);
    L1(iter)=L1_H;
    cost=L(iter)+L1(iter);
    dcost=cost_old-cost;
    cost_old=cost;
    disp(sprintf('%5d | %10.2f | %10.6g | %11.2f | %10.6g | %10.6g ', iter,cost,dcost/cost,(1-(SST+L_H)/SST)*100,mu_W,mu_H))    
end
Predmis=dprod(M,W,H);
varExp=(1-(SST+L_H)/SST)*100;

%---------------------------------------------
function [W,mu,cost]=NI_PG(XHt,HHt,W,H,I,J,mu,nonneg,lambda)

maxiter=10;
[Rmis,valmis]=dprodIJ(I,J,W,H);
costold=sum(sum((W'*W).*(HHt)))-sum(valmis.^2)-2*sum(sum(W.*XHt));
for iter=1:maxiter       
   G=W*HHt-Rmis*H'-XHt;     
   if lambda>0
    G=G-bsxfun(@times,sum(G.*W,1),W);
   end
   Wold=W;
   stop=0;
   while ~stop
      W=Wold-mu*G;
      if nonneg
        W(W<0)=0;           
      end
      if lambda>0
        W=bsxfun(@times,W,1./bsxfun(@max,sqrt(sum(W.^2,1)),eps));
      end      
      [Rmis,valmis]=dprodIJ(I,J,W,H);           
      cost=sum(sum((W'*W).*(HHt)))-sum(valmis.^2)-2*sum(sum(W.*XHt));
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
function [H,mu,L,L1]=SC_PG(WtX,WtW,W,H,lambda,I,J,mu,nonneg)

maxiter=10;
[Rmis,valmis]=dprodIJ(I,J,W,H);
costold=0.5*(sum(sum((WtW).*(H*H')))-sum(valmis.^2)-2*sum(sum(H.*WtX)))+lambda*sum(sum(abs(H)));

for iter=1:maxiter
   G=WtW*H-W'*Rmis-WtX;    
   Hold=H;
   stop=0;
   while ~stop
      H=Hold-mu*G;      
      H(abs(H)<mu*lambda)=0;
      H=H-mu*lambda*sign(H);            
      if nonneg
        H(H<0)=0;           
      end
      [Rmis,valmis]=dprodIJ(I,J,W,H);      
      L=sum(sum((WtW).*(H*H')))-sum(valmis.^2)-2*sum(sum(H.*WtX));
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


%--------------------------------------------------------------------
function [R,val]=dprodIJ(I,J,W,H)

W=W';
step=10000;
indt=1:step;
val=zeros(1,length(I));
for k=1:ceil((length(I)/step))
    ind=(k-1)*step+indt;
    if ind(end)>length(I)
       t=find(ind==length(I));
       ind=ind(1:t);
    end
    val(ind)=sum(W(:,I(ind)).*H(:,J(ind)),1);
end
R=sparse(I,J,val,size(W,2),size(H,2));



%--------------------------------------------------------------------
function [R,val]=dprod(M,W,H)

W=W';
[I,J]=find(M);
step=10000;
indt=1:step;
val=zeros(1,length(I));
for k=1:ceil((length(I)/step))
    ind=(k-1)*step+indt;
    if ind(end)>length(I)
       t=find(ind==length(I));
       ind=ind(1:t);
    end
    val(ind)=sum(W(:,I(ind)).*H(:,J(ind)),1);
end
R=sparse(I,J,val,size(M,1),size(M,2));
