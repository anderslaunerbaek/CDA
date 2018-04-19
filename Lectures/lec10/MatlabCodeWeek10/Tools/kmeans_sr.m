function [C,S,Q,progress]=kmeans_sr(X,noc,opts)

% The sr-clustering algortihm for graph clustering described in 
% M. Mørup and L.K.Hansen "An Exact Relaxation of the Hard Assignment
% Problem in Clustering", submitted to Journal of Machine Learning Research
% 2009.
%
% If X is a I x J matric where J index over observations the algorithm optimizes the k-means objective: 
%   minimize ||X-CS||_F^2 s.t. |S_i|_1=1, S_{d,i}>=0,
%   this is equivalent to maximizing 
%   Q=trace[SXtXS'D] where D=diag(1./sum(S.^2,2))
% If X is a J x J symmetric similarity matrix the objective is given by 
%   Q=trace[SXS'D] where D=diag(1./sum(S.^2,2))
%
% usage:
%   [C,S,Q,progress]=sr_clustering_pc(X,noc,opts)
%   
% Input:
%   X       I x J  matrix, if I ~= J, X are points in R^I 
%                          if I==J, X is a similarity matrix
%   noc     number of clusters
%   opts.
%           maxiter     maximum number of iterations
%           S           I x noc matrix of community identities
%           mu_s        initial stepsize (default 1/m)
%           verbose     
% Output:
%   Q        value of the k-means objective function, Q=trace[SXtXS'D] 
%   S        Cluster indicator matrix of size d x I where d<=noc
%   progress iteration performance

if nargin<3
    opts=struct;
end
[I,J]=size(X);
sym=0;
if I==J 
    sym=(norm(X-X','fro')==0);
    XtX=X;
    SST=trace(XtX);
    objstring='trace(SXS^TD)';
elseif I>J
   XtX=X'*X; 
   SST=trace(XtX);
   objstring='trace(SX^TXS^TD)';
else
    SST=sum(sum(X.^2));
    objstring='trace(SX^TXS^TD)';
end
% Initialize variables
warning('off', 'MATLAB:divideByZero');
S=mgetopt(opts,'S',rand(noc,J));
maxiter=mgetopt(opts,'maxiter',500);
S=S./repmat(sum(S),noc,1);
mu_s=mgetopt(opts,'mu_s',0.01);
verbose=mgetopt(opts,'verbose',1);
D=repmat(1./sum(S,2),1,J);
progress=zeros(1,maxiter+1);
if I>J || sym
    SXtXD=(S.*D)*XtX;
else
    SXtXD=((S.*D)*X')*X;
end
Q=sum(sum(SXtXD.*S));
dQ=inf;
progress(1)=SST-Q;
iter=0;
Cest=X*S'*diag(1./sum(S,2));

if verbose % Display algorithm
    disp(['Performing pairwise clustering based on ' objstring]);
    dheader = sprintf('%12s | %12s | %12s | %12s | %12s | %12s','Iteration','% var. expl',objstring,'dQ/Q','step size','time');
    dline = sprintf('-------------+--------------+--------------+--------------+--------------+--------------');
    disp(dline);
    disp(dheader);
    disp(dline);
end
while dQ>Q*1e-10 && iter<maxiter
    iter=iter+1;
    tic;
    g=2*SXtXD-repmat(sum(SXtXD.*(S.*D),2),1,J);
    g=g-repmat(sum(g.*S),noc,1);

    Sold=S;
    stop=0;
    while ~stop % perform projected L1 constrained gradient ascent
        S=Sold+mu_s*g/max(abs(g(:)));
        S(S<0)=0;
        S=S./repmat(sum(S)+eps,noc,1);
        D=repmat(1./(sum(S,2)+eps),1,J);
        if I>J || sym
            SXtXD=(S.*D)*XtX;
        else
            SXtXD=((S.*D)*X')*X;
        end
        Qnew=sum(sum(SXtXD.*S));
        dQ=Qnew-Q;
        if dQ>=0 || mu_s<1e-12
            mu_s=mu_s*2;
            stop=1;
            Q=Qnew;
        else
            if verbose
                disp('Checking stationarity')
            end
            mu_s=mu_s/100;
        end
    end
    progress(iter+1)=Q;
    if rem(iter,10)==0 && verbose
        disp(sprintf('%12.0f | %12.4f | %12.4e  | %12.4e | %12.4e | %12.4f ',iter,Q/SST,Q,dQ/Q,mu_s,toc));
    end   
end
if verbose
     disp(sprintf('%12.0f | %12.4f | %12.4e  | %12.4e | %12.4e | %12.4f ',iter,Q/SST,Q,dQ/Q,mu_s,toc));
end
progress(progress==0)=[];

% Remove empty cluster
[val,l]=sort(sum(S,2),'descend');
S=S(l,:);
S(val==0,:)=[];
if I~=J 
    C=X*S'*diag(1./sum(S,2));
else
    C=[];
end
   
% -------------------------------------------------------------------------
% Parser for optional arguments
function var = mgetopt(opts, varname, default, varargin)
if isfield(opts, varname)
    var = getfield(opts, varname); 
else
    var = default;
end
for narg = 1:2:nargin-4
    cmd = varargin{narg};
    arg = varargin{narg+1};
    switch cmd
        case 'instrset',
            if ~any(strcmp(arg, var))
                fprintf(['Wrong argument %s = ''%s'' - ', ...
                    'Using default : %s = ''%s''\n'], ...
                    varname, var, varname, default);
                var = default;
            end
        otherwise,
            error('Wrong option: %s.', cmd);
    end
end

