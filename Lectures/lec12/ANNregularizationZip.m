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
hiddenLayerSize = [?? ??]; % <-- Test different zero, one and two layer networks

net = patternnet(hiddenLayerSize);

% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 50/100;
net.divideParam.valRatio   = 0/100; % Don't use validation set. Iterate till convergence.
net.divideParam.testRatio  = 50/100;


net.performParam.regularization = ??; % <-- Define regularization,parameter in 0 - 1 

net.trainParam.epochs = 200;

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

