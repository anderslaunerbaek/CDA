function plotTucker(G,A,B,C,EmAx,ExAx)
% Plots the factors of the claus.mat data based on the PARAFAC model
for kp=1:size(A,2)
    figure('name',['mode 1 component ' num2str(kp)]);
    plotG(G,A,B,C,EmAx,ExAx,kp);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plotG(G,A,B,C,EmAx,ExAx,kp);

h = gcf;
clf;
set(h, 'Color', [1 1 1]);
set(h, 'PaperUnits', 'centimeters');
set(h, 'PaperType', '<custom>');
set(h, 'PaperPosition', [0 0 12 12]);

d1=size(C,2);
d2=size(B,2);
air=0.4; % percent air between plots;
s1=(d1+1)+(d1+2)*air;
s2=(d2+1)+(d2+2)*air;

s_d1=1/s1;
s_d2=1/s2;
s_air1=air/s1;
s_air2=air/s2;

h = axes('position', [0 0 1 1]);
set(h, 'Visible', 'off');

h = axes('position', [s_air1,d2*s_air2+d2*s_d2, s_d1, s_d2]);
bar(A(:,kp))
axis tight;
set(get(h,'Title'),'String',['A(:,' num2str(kp) ')' ],'FontWeight','bold')

axis off;
for k=1:size(B,2)
    h = axes('position', [s_air1, k*s_air2+(k-1)*s_d2, s_d1, s_d2]);
    plot(EmAx,B(:,size(B,2)-k+1))
    axis tight;
    set(get(h,'Ylabel'),'String',['B(:,' num2str(size(C,2)-k+1) ')' ],'FontWeight','bold')
end

for k=1:size(C,2)
    h = axes('position', [k*s_d1+(k+1)*s_air1, d2*s_air2+d2*s_d2, s_d1, s_d2]);
    plot(ExAx,C(:,k));
    axis tight;
    set(get(h,'Title'),'String',['C(:,' num2str(k) ')' ],'FontWeight','bold')
end
h = axes('position', [2*s_air1+s_d1,s_air2,(d1-2)*s_air1+d1*s_d1, (d2-2)*s_air2+d2*s_d2]);
hinton(squeeze(G(kp,:,:)),[],[],max(abs(G(:))));
set(get(h,'Title'),'String',['Tucker Core G(' num2str(kp) ',:,:)'],'FontWeight','bold')

