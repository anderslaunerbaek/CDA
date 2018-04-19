function coreconsistency=CorConDiag(X,A,B,C)
% Core Consistency Diagnostic for a 3-way array
% The core consistency measures deviation between PARAFAC and Tucker core
% array based on a Tucker core defined by the PARAFAC loadings.
%
% Usage:
%   coreconsistency=CorConDiag(X,A,B,C)
%
% Input:
%   X             3-way array to decompose
%   A             the loadings for the first mode
%   B             the loadings for the second mode
%   C             the loadings for the third mode
%
% Output:
%   coreconsistency  Value of the core consistency diagnostic


noc=size(A,2);
I=zeros(noc,noc,noc);
for j=1:noc
    I(j,j,j)=1;
end
G=tmult(tmult(tmult(X,pinv(A),1),pinv(B),2),pinv(C),3);

coreconsistency=100*(1-sum((G(:)-I(:)).^2)/noc);

if coreconsistency<0
    coreconsistency=0;
end

    