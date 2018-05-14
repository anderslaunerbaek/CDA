close all
clear all

load zipdata

% Transform to ten element logical vector
Y = zeros(400,10);
for i=1:400
    Y(i,y(i)+1) = 1;
end

X = X';
Y = Y';

% Create a Pattern Recognition Network
hiddenLayerSize = [??]; % <-- Test different zero, one and two layer networks

net = patternnet(hiddenLayerSize);

% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 50/100;
net.divideParam.valRatio   = 25/100;
net.divideParam.testRatio  = 25/100;

% Train the Network
[net,tr] = train(net,X,Y);

% View the Network
view(net)

% Plot
figure, plotperform(tr)

% Test data evaluation
tInd  = tr.testInd;
Xtest = X(:,tInd);
Ytest = Y(:,tInd);
Yref  = y(tInd);

% Calculate test data predictions 
Yhat  = net(Xtest);

figure
plotconfusion(Ytest,Yhat)

% Show output
[~, Ypred] = max(Yhat);
Ypred = Ypred' - 1;
figure
for i=1:size(Xtest,2)
    clf
    subplot(221)
    c=reshape(Xtest(:,i),16,16);
    c=rot90(c,1);
    c = [c;zeros(1,16)]; %pcolor does not show last row and column
    c = [c zeros(17,1)];
    pcolor(-c)
    caxis([0 .5])
    set(gca,'xtick',[],'ytick',[])
    axis square
    title(['Predicted ',num2str(Ypred(i)),', actual ',num2str(Yref(i))])
    subplot(212)
    bar((0:9),Yhat(:,i))
    axis([-1 10 0 1])
    title('Digit recognition probability')
    pause
end
