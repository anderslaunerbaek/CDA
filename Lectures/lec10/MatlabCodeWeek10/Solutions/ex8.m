clear all;
close all;
load zipdata;

y=traindata(:,1)'+1;
X=traindata(:,2:end)';
X=X-min(min(X));
Ndigit=250;
noc=25;

disp('running NMFLS algorithm')
[W_LS,H_LS,L_LS]=NMFLS(X(:,1:Ndigit),noc,250);

% Start in LS solution and optimize for KL
disp('running NMFKL algorithm from LS solution')
[W_KL,H_KL,L_KL,L_LSKL]=NMFKL(X(:,1:Ndigit),noc,250,W_LS,H_LS);

% Display LS error 
figure;
hold on;
subplot(1,2,1);
plot(L_LS,'.-r');
axis tight;
title('NMFLS optimization','FontWeight','bold');
ylabel('Objective value','FontWeight','bold');
xlabel('iteration number','FontWeight','bold');

% Display KL error and corresponding LS error
subplot(1,2,2);
hold on;
plot(L_KL,'.-');
plot(L_LSKL,'.-r'); % Corresponding LS error for each iteration of NMFKL
axis tight;
title('NMFKL optimization','FontWeight','bold');
ylabel('Objective value','FontWeight','bold');
xlabel('iteration number','FontWeight','bold');
legend({'KL-divergence','LS-error'});

% Display the extracted features of NMFLS
figure;
subplot(1,2,1);
montageplot(permute(reshape(W_LS,[16,16,noc]),[3 2 1]));
title(['Features extracted by NMF LS'],'FontWeight','bold');
colormap(1-gray);
axis equal;
axis off;

% Display the extracted features of NMFKL
subplot(1,2,2);
montageplot(permute(reshape(W_KL,[16,16,noc]),[3 2 1]));
title(['Features extracted by NMF KL'],'FontWeight','bold');
colormap(1-gray);
axis equal;
axis off;

