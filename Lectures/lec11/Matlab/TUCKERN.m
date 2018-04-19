function [Core, FACT] = TUCKERN(X, D)

% Tucker optimization based on alternating least squares
% Model:
%
% X_{i1,i2,...,in}=\sum_j1,j2,...,jn Core_j1,j2...,jn*FACT_{i1,j1}*FACT_{i2,j2}*...*FACT_{in,jn}
%
% Usage:
% [Core, FACT]=CP(X, d)
%
% Input:
% X             n-way array to decompose
% D             vector of number of components of each mode
%
% Output:
% Core          The Tucker core array
% FACT          cell array: FACT{i} is the factors found for the i'th
%               modality

Nx=ndims(X);
N=size(X);

% Initialize FACT
FACT=cell(1,Nx);
G=X;
for i=1:Nx
   FACT{i}=rand(N(i),D(i)); 
   G=tmult(G,FACT{i}',i);
end

% Set algorithm parameters
iter=0;
SST=sum(X(:).^2);
SSE=inf;
dSSE=inf;
tic;
        

disp([' '])
disp(['Tucker optimization'])
disp(['A ' num2str(D) ' component model will be fitted'])
dheader = sprintf('%12s | %12s | %12s | %12s |','Iteration','Expl. var.','dSSE','Time');
dline = sprintf('-------------+--------------+--------------+--------------+');

while dSSE>=1e-9*SSE && iter<1000 

        if mod(iter,100)==0
             disp(dline); disp(dheader); disp(dline);
        end
        iter=iter+1;
        SSE_old=SSE;
        
        % Estimate Factors        
        for i=1:Nx
            NN=1:Nx;
            NN(i)=[];
            Xt=X;
            for j=NN
                Xt=tmult(Xt,FACT{j}',j);            
            end
            Xt=matricizing(Xt,i);
            [U,S,V]=???
            FACT{i}=???            
        end
        Gn=FACT{Nx}'*Xt;        
        
        SSE=SST-sum(sum(Gn(:).^2));
        dSSE=SSE_old-SSE;

        % Display iteration
        if rem(iter,1)==0
            disp(sprintf('%12.0f | %12.4f | %6.5e | %12.4e ',iter, (SST-SSE)/SST,dSSE,toc));
            tic;
        end
end
% Display final iteration
disp(sprintf('%12.0f | %12.4f | %6.5e | %12.4e ',iter, (SST-SSE)/SST,dSSE,toc));
Core=unmatrizicing(Gn,Nx,D);

    