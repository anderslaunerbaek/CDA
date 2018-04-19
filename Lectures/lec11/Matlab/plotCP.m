function plotCP(A,B,C,EmAx,ExAx,y)
% Plots the factors of the claus.mat data based on the CP model

noc=size(A,2);
figure; 
for d=1:noc
    subplot(noc,3,(d-1)*3+1)
    [val,ind]=max(abs(A(:,d)));
    if A(ind,d)<0 % Flip sign if negative
        A(:,d)=-A(:,d);
        B(:,d)=-B(:,d);
    end    
    bar(A(:,d));    
    axis tight;
    if d==1
       title('Estimated sample concentration','fontweight','bold') 
    end
    ylabel(['Component ' num2str(d)],'fontweight','bold' )
        
    subplot(noc,3,(d-1)*3+2)
    [val,ind]=max(abs(B(:,d)));
    if B(ind,d)<0 % Flip sign if negative
        B(:,d)=-B(:,d);
        C(:,d)=-C(:,d);
    end
    plot(EmAx,B(:,d),'linewidth',2);
    xlabel('nm','fontweight','bold')
    if d==1
       title('emission spectra','fontweight','bold') 
    end
    axis tight;
    
    subplot(noc,3,(d-1)*3+3)
    plot(ExAx,C(:,d),'linewidth',2);
    xlabel('nm','fontweight','bold')
    if d==1
       title('excitation spectra','fontweight','bold') 
    end
    axis tight;
end

figure; 
for d=1:size(y,2)
    subplot(3,1,d)
    bar(y(:,d));
    if d==1;
        title('True concentrations','fontweight','bold');
    end
    ylabel(['Compound ' num2str(d)],'fontweight','bold' )
end
    