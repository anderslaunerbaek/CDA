clear all;
close all;

load IMAGES;

patch_size=[16 16];
Ncol=size(IMAGES,2)/patch_size(2);
Nrow=size(IMAGES,1)/patch_size(1)
Nim=size(IMAGES,3);

% Patch images with no overlap between patches
X=zeros(prod(patch_size),Nim*Nrow*Ncol);
for im=1:Nim
    for i=1:Nrow
        for j=1:Ncol
            patch=IMAGES((i-1)*patch_size(1)+1:i*patch_size(1),(j-1)*patch_size(2)+1:j*patch_size(2),im);
            X(:,(im-1)*Nrow*Ncol+(i-1)*Ncol+j)=patch(:)';
        end
    end
end

noc=49;
lambda=0.5;
Npatches=2500;

%% Sparse Coding Analysis
[W_SC,H_SC,L,L1]=SparseCoding(X(:,1:Npatches),noc,lambda);    

% Plot W
figure(1);  
subplot(2,2,1);
montageplot(permute(reshape(W_SC,[patch_size,noc]),[3 2 1])); 
title(['Estimated W for sparse coding, lambda=' num2str(lambda)]);
colormap(gray); 
axis equal; 
axis tight;
axis off;

% Plot H
subplot(2,2,2);   
hist((H_SC(:)),100)
ylabel('Number of elements in H','FontWeight','bold')
xlabel('Bin value','FontWeight','bold');
title(['Histrogram of the values in H for sparse coding, lambda=' num2str(lambda)]); 

%%
% SVD analysis
[U,S,V]=svd(X(:,1:Npatches),'econ');    
W_SVD=U(:,1:noc);
H_SVD=S(1:noc,1:noc)*V(:,1:noc)';

% Plot W
subplot(2,2,3);   
montageplot(permute(reshape(W_SVD,[patch_size,noc]),[3 2 1])); 
title(['Estimated W by SVD']);
colormap(gray); 
axis equal; 
axis tight;
axis off;

% Plot H
subplot(2,2,4); 
hist((H_SVD(:)),100)
ylabel('Number of elements in H','FontWeight','bold')
xlabel('Bin value','FontWeight','bold');
title(['Histrogram of the values in H for SVD']);        



