methods={'SVD','NMF', 'AA', 'SC', 'NSC', 'ICA', 'kmeans'}
method=methods{2};

noc=2;      % For SVD the number of components (noc) can maximally be set to 2
lambda=.01; % L1 regularization strength for the methods SC and NSC

load Synthdata;

figure;
for k=1:length(X)
    subplot(2,3,k)
    hold on;
    plot(X{k}(1,:),X{k}(2,:),'.');   
    plot([0 0],[ -2 2],'k-','linewidth',2);
    plot([-2 2],[0 0],'k-','linewidth',2);
    axis equal;
    axis([-2 2 -2 2])
    
    % Carry out analysis
    SST=sum(sum(X{k}.^2));
    switch method
        case 'SVD'
             [U,S,V]=svd(X{k},'econ');
             U=U(:,1:noc);
             S=S(1:noc,1:noc);
             V=V(:,1:noc);
             W=U;
             H=S*V';
             L=0.5*(SST-trace(S.^2));
             for t=1:min([size(W,2),noc])
                plot([0,W(1,t)],[0,W(2,t)],'-r','LineWidth',2');
             end
        case 'AA'
             [W,S,H,L]=ArchetypalAnalysis(X{k},noc);             
             plot(W(1,[1:noc 1]),W(2,[1:noc 1]),'-r','LineWidth',2');               
        case 'NMF'
             [W,H,L]=NMFPG(X{k},noc);
             for t=1:noc
                plot([0,W(1,t)],[0,W(2,t)],'-r','LineWidth',2');
             end                       
        case 'SC'
             [W,H,L,L1]=SparseCoding(X{k},noc,lambda,[0 0]);
             for t=1:noc
                plot([0,W(1,t)],[0,W(2,t)],'-r','LineWidth',2');
             end            
        case 'NSC'
             [W,H,L,L1]=SparseCoding(X{k},noc,lambda,[1 1]);
             for t=1:noc
                plot([0,W(1,t)],[0,W(2,t)],'-r','LineWidth',2');
             end             
        case 'kmeans'
             [W,H]=kmeans_sr(X{k},noc);
             for t=1:noc
                plot(W(1,t),W(2,t),'*r','MarkerSize',10);
             end
             L=0.5*sum(sum((X{k}-W*H).^2));
        case 'ICA'
            [H, W, Winv] = FASTICA (X{k},'numOfIC',noc,'approach','symm','g','tanh');
            % normalize components
            d=sqrt(sum(W.^2));
            W=W*diag(1./d);
            H=diag(d)*H;
            L=0.5*sum(sum((X{k}-W*H).^2));
            for t=1:min([noc,size(W,2)])
                plot([0,W(1,t)],[0,W(2,t)],'-r','LineWidth',2');
            end                        
    end
    title(['Analyzed by ' method ', VE=' num2str(1-2*L(end)/SST)],'FontWeight','bold' )       
end