methods={'SVD','NMF', 'AA', 'SC', 'NSC', 'kmeans'}
method=methods{1};

noc=49; % Number of components
K=5;    % Number of nearest neighbours used for classification

lambda=.1; % L1 regularization used for 'SC' and 'NSC'

Ntrain=2000; % Use first Ntrain observations of the training data for training (Ntrain can maximally be 7291 )
Ntest=500;   % Use first Ntest observations of the test data for testing (Ntest can maximally be 2007)
load zipdata;

y=traindata(1:Ntrain,1)'+1;
X=traindata(1:Ntrain,2:end)';
ytest=testdata(1:Ntest,1)'+1;
Xtest=testdata(1:Ntest,2:end)';
minX=min(min(X));
X=X-minX; % Make sure X is non-negative (required for NMF)
Xtest=Xtest-minX; 

SST=sum(sum(X.^2));
disp(['Performing dimensionality reduction using ' method]);
switch method
        case 'SVD'             
             [U,S,V]=svd(X,'econ');                         
             W=U(:,1:noc);
             H=S(1:noc,1:noc)*V(:,1:noc)';
             L=0.5*(SST-trace(S(1:noc,1:noc).^2));          
        case 'AA'
             [W,S,H,L]=ArchetypalAnalysis(X,noc);                          
        case 'NMF'
             [W,H,L]=NMFPG(X,noc);             
        case 'SC'
             [W,H,L,L1]=SparseCoding(X,noc,lambda,[0 0]);             
        case 'NSC'
             [W,H,L,L1]=SparseCoding(X,noc,lambda,[1 1]);           
        case 'kmeans'
             [W,H]=kmeans_sr(X,noc);             
             L=0.5*sum(sum((X-W*H).^2));         
end

if length(L)>1
    figure('name',['iteration vs. objective value for ' method])
    plot(L,'o-');
    xlabel('Iteration','FontWeight','bold');
    ylabel('$0.5\|\mathbf{X}-\mathbf{WH}\|_F^2$','Interpreter','latex','FontWeight','bold');    
end

% Plot extrated features and calculate errorRate
figure('name',[method ' features']); 
montageplot(permute(reshape(W,[16,16,noc]),[3 2 1]));
title(['Variation explained by '  method ' ' num2str(1-2*L(end)/SST)]);
colormap(1-gray);
axis equal;
axis off;

% Plot the first Nplot reconstructed digits from their extracted sub-space representation
disp('Performing nearest neighbour classification in subspace');
Htest=projectTestdata(W,Xtest,method);
errorRate=evaluateKNN(H,y',Htest,ytest',K);
Nplot=100; % Number of digits to reconstruct for display
figure('name',['reconstructed ' num2str(Nplot) ' first digits using ' method ]); 
montageplot(permute(reshape(W*H(:,1:Nplot),[16,16,Nplot]),[3 2 1]));
title(['Error rate based on ' method ' ' num2str(errorRate)]);
colormap(1-gray);
axis equal;
axis off;


