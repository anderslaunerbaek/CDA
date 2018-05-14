function [rmse, cputime] = ANNcylinder(hiddenLayerSize)
% 
% hiddenLayerSize = N, for one hidden layer
% hiddenLayerSize = [N M], for two hidden layers
%
tic
    n = 200;
    [X, Y] = meshgrid(linspace(-3,3,n),linspace(-3,3,n));
    Z = real(X.^2 + Y.^2 <= 1);

    figure
    clf
    surf(X,Y,Z,'FaceAlpha',.75,'EdgeColor','none','FaceColor',[.7 .7 .7],'facelighting','gouraud','AmbientStrength',0.7)
    hold on

    % Create a Fitting Network
    net = fitnet(hiddenLayerSize);
    
    % Set up Division of Data for Training, Validation, Testing
    net.divideParam.trainRatio = 0.5;
    net.divideParam.valRatio   = 0.25;
    net.divideParam.testRatio  = 0.25;
    
    inputs  = [X(:) Y(:)]';
    targets = Z(:)';
    
    % Train the Network
    [net,tr] = train(net,inputs,targets);
 
    n = 30;
    [X2, Y2] = meshgrid(linspace(-3,3,n),linspace(-3,3,n));
    Z2 = real(X2.^2 + Y2.^2 <= 1);
    inputs2 = [X2(:) Y2(:)]';
    
    % Test the Network
    outputs2 = net(inputs2);
    
    Zhat = reshape(outputs2,n,n);
    
    surf(X2,Y2,Zhat,'EdgeColor','r','FaceColor','none')

    light('Position',[-8 -8 .5],'Style','local')
    axis([-3 3 -3 3 -.5 1.5]) 
    rmse = sqrt(mean((Zhat(:)-Z2(:)).^2)) / sqrt(mean((Z2(:)-mean(Z2(:))).^2));
    
    if length(hiddenLayerSize)==1
        title(['One layer, ',num2str(hiddenLayerSize), ' hidden nodes, rRMSE = ',num2str(rmse)])
    else
        title(['Two layers, ',num2str(hiddenLayerSize(1)),' - ',num2str(hiddenLayerSize(2)),' hidden nodes, rRMSE = ',num2str(rmse)])
    end
cputime = toc;
   
    
    