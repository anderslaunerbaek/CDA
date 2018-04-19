clear all;
load MixedSound.mat;

%%
% Listen to the mixed sounds
channel=1;
soundsc(X(channel,:),Fs);

%% Analyze the data
noc=3;
%[W,S,H,L]=ArchetypalAnalysis(X,noc);
% [W,H,L]=NMFPG(X,noc);
% [H, W, Winv] = FASTICA (X,'numOfIC',noc,'approach','symm','g','tanh');
 [U, S, V] = svd(X,'econ'); W=U(:,1:noc); H=S(1:noc,1:noc)*V(:,1:noc)';
% [W,H,L,L1]=SparseCoding(X,noc,0.1,[0 0]);
% [W,H]=kmeans_sr(X,noc);

%%
% Listen to the result
comp=1;
soundsc(H(comp,:),Fs);

