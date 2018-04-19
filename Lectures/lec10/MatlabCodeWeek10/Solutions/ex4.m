clear all;
load NMR_mix_DoEcompressed.mat;

noc=3;
[W,S,H,L]=ArchetypalAnalysis(xData,noc);
% [W,H,L]=NMFPG(xData,noc);
% [H, W, Winv] = FASTICA (xData,'numOfIC',noc,'approach','symm','g','tanh');
% [U, S, V] = svd(xData,'econ'); W=U(:,1:noc); H=S(1:noc,1:noc)*V(:,1:noc)';
% [W,H,L,L1]=SparseCoding(xData,noc,0.1,[0 0]);
% [W,H]=kmeans_sr(xData,noc);

figure;
subplot(2,2,1);
imagesc(H);
colorbar;
title('Estimated concentrations','FontWeight','bold');

subplot(2,2,2);
imagesc(yData);
colorbar;
title('True concentrations','FontWeight','bold');

subplot(2,2,3:4);
hold all;
for k=1:size(W,2)
    plot(Axis,W(:,k));
end
title('Component specific spectral profiles','FontWeight','Bold');
legend({'Component 1','Component 2','Component 3'})
axis tight;
