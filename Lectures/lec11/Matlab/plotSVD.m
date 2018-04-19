function plotSVD(A,B,EmAx,ExAx);

% Plots claus data result obtained from SVD decomposition where 
% A=U
% B=S*V'

noc=size(A,2);

for d=1:noc
    subplot(noc,2,(d-1)*2+1)
    [val,ind]=max(abs(A(:,d)));
    if A(ind,d)<0 % Flip sign if negative
        A(:,d)=-A(:,d);
        B(d,:)=-B(d,:);
    end
    bar(A(:,d));
    axis tight;
    ylabel(['Component ' num2str(d)],'fontweight','bold' )
    if d==1
        title('Estimated sample concentration','fontweight','bold') 
    end
    
    subplot(noc,2,(d-1)*2+2)
    imagesc(ExAx,EmAx,reshape(B(d,:),length(EmAx),length(ExAx)));
    if d==1
        title('emission x excitation spectra','fontweight','bold')
    end
    xlabel('nm','Fontweight','bold')
    ylabel('nm','Fontweight','bold')
    axis tight;            
end