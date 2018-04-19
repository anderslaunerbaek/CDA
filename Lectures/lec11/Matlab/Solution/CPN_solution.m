function FACT=CP(X,d)

% The CandeComp/PARAFAC 
%
% X_{i1,i2,...,in}=\sum_k FACT{1}_{i1,k}*FACT{2}_{i2,k}*...*FACT{n}_{in,k}
%
% Usage:
% FACT=CP(X,d)
%
% Input:
% X             n-way array to decompose
% d             number of components
%
% Output:
% FACT          cell array: FACT{i} is the factors found for the i'th
%               modality

Nx=ndims(X);
N=size(X);

% Initialization
for i=1:Nx       
    FACT{i}=randn(N(i),d);
end

SSE=inf;
SST=sum(X(:).^2);
dSSE=inf;
maxiter=250;

disp([' '])
disp(['CP optimization'])
disp(['A ' num2str(d) ' component model will be fitted']);
disp([' '])
disp(['To stop algorithm press control C'])
disp([' ']);
dheader = sprintf('%12s | %12s | %12s | %12s ','Iteration','Expl. var.','dSSE','Time');
dline = sprintf('-------------+--------------+--------------+--------------+');

tic;
iter=0;
while dSSE>=1e-6*SSE & iter<maxiter
        if mod(iter,100)==0 
             disp(dline); disp(dheader); disp(dline);
        end
        iter=iter+1;
        SSE_old=SSE;
        for i=1:Nx                        
             ind=1:Nx;
             ind(i)=[];                
             kr=FACT{ind(1)};
             krkr=FACT{ind(1)}'*FACT{ind(1)};
             for z=ind(2:end)
                 kr=krprod(FACT{z}, kr);     
                 krkr=krkr.*(FACT{z}'*FACT{z});
             end
             Xkr=matricizing(X,i)*kr;
             FACT{i}=Xkr/krkr;
        end
        SSE=sum(sum(krkr.*(FACT{i}'*FACT{i})))-2*sum(sum(Xkr.*FACT{i}))+SST;
        dSSE=SSE_old-SSE;
        if mod(iter,5)==0 
            disp(sprintf('%12.0f | %12.4f | %12.4e | %12.4e |',iter, (SST-SSE)/SST,dSSE,toc));
            tic;
        end        
end
% Display final iteration
disp(sprintf('%12.0f | %12.4f | %12.4e | %12.4e |',iter, (SST-SSE)/SST,dSSE/SSE,toc));

