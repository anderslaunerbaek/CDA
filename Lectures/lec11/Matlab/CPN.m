function FACT=CPN(X,d)

% The CandeComp/PARAFAC model optimized by alternating least squares
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

addpath('N-way-tools');
Nx=ndims(X);
N=size(X);

% Initialization
for i=1:Nx       
    FACT{i}=???????
end

SSE=inf;
SST=sum(X(:).^2);
dSSE=inf;
maxiter=1000;
tic;
iter=0;

disp([' '])
disp(['CP optimization'])
disp(['A ' num2str(d) ' component model will be fitted']);
dheader = sprintf('%12s | %12s | %12s | %12s ','Iteration','Expl. var.','dSSE','Time');
dline = sprintf('-------------+--------------+--------------+--------------+');

while dSSE>=1e-6*SSE & iter<maxiter
        if mod(iter,100)==0 
             disp(dline); disp(dheader); disp(dline);
        end
        iter=iter+1;
        SSE_old=SSE;

        for i=1:Nx                        
            % Update factors making use of the functions 
            %   matrizicing.m 
            %   krprod.m
            ????
            ????
             FACT{i}=??????????
        end
        % Sum of Square Error
        SSE=????????
        dSSE=SSE_old-SSE;
        
        if mod(iter,5)==0 
            disp(sprintf('%12.0f | %12.4f | %12.4f | %12.4e |',iter, (SST-SSE)/SST,dSSE,toc));
            tic;
        end        
end
% Display final iteration
disp(sprintf('%12.0f | %12.4f | %12.4f | %12.4e |',iter, (SST-SSE)/SST,dSSE,toc));

