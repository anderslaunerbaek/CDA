function [A,B,C]=CP(X,D)

% The CandeComp/PARAFAC model optimized by alternating least squares for a
% 3-way array
%
% X_ijk=\sum_d A_id B_jd C_kd
%
% Usage:
% FACT=CP(X,D)
%
% Input:
% X             n-way array to decompose
% D             number of components
%
% Output:
% A             First mode loadings
% B             Second mode loadings
% C             Third mode loadings

N=size(X);

% Initialization
A=randn(N(1),D);
B=randn(N(2),D);
C=randn(N(3),D);

SSE=inf;
SST=sum(X(:).^2);
dSSE=inf;
maxiter=250;
tic;
iter=0;

disp([' '])
disp(['3-way CP optimization'])
disp(['A ' num2str(D) ' component model will be fitted']);
dheader = sprintf('%12s | %12s | %12s | %12s ','Iteration','Expl. var.','dSSE','Time');
dline = sprintf('-------------+--------------+--------------+--------------+');

while dSSE>=1e-9*SSE && iter<maxiter
        if mod(iter,100)==0 
             disp(dline); disp(dheader); disp(dline);
        end
        iter=iter+1;
        SSE_old=SSE;
        
        % Update factors making use of the functions 
        %   matrizicing.m 
        %   krprod.m
        A=???
        B=???
        C=???

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

