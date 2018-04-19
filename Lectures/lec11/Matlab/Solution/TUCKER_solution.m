function [G, A, B, C] = TUCKER(X, D)

% Tucker optimization based on alternating least squares
% Model:
%
% X_ijk=\sum_lmn G_lmn*A_il*B_jm*C_kn
%
% Usage:
% [G,A,B,C]=Tucker(X, D)
%
% Input:
% X             n-way array to decompose
% D             1 x 3 vector of number of components for each mode
%
% Output:
% G             The Tucker core array
% A             the loadings for the first mode
% B             the loadings for the second mode
% C             the loadings for the third mode

Nx=ndims(X);
N=size(X);

% Initialize A,B,C by random
A=randn(N(1),D(1));
B=randn(N(2),D(2));
C=randn(N(3),D(3));

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

while dSSE>=1e-9*SSE && iter<250 

        if mod(iter,100)==0
             disp(dline); disp(dheader); disp(dline);
        end
        iter=iter+1;
        SSE_old=SSE;
                                
        % Estimate A,B,C        
        [U,S,V]=svd(matricizing(X,1)*kron(C,B),0);
        A=U(:,1:D(1));
        
        [U,S,V]=svd(matricizing(X,2)*kron(C,A),0);
        B=U(:,1:D(2));
        
        [U,S,V]=svd(matricizing(X,3)*kron(B,A),0);
        C=U(:,1:D(3));
                
        % Estimate Core
        G=tmult(tmult(tmult(X,A',1),B',2),C',3);

        % Evaluate least squares error        
        SSE=SST-sum(sum(sum(G.^2)));
        dSSE=SSE_old-SSE;

        % Display iteration
        if rem(iter,1)==0
            disp(sprintf('%12.0f | %12.4f | %6.5e | %12.4e ',iter, (SST-SSE)/SST,dSSE,toc));
            tic;
        end
end
% Display final iteration
disp(sprintf('%12.0f | %12.4f | %6.5e | %12.4e ',iter, (SST-SSE)/SST,dSSE,toc));
    