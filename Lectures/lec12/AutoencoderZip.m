function AutoencoderZip

close all

load zipdata

% Separate training and test data
rng(0,'twister'); % Same training sequence every time
Itrain = rand(400,1)>.25;
Xtrain = X(Itrain,:)';
Xtest  = X(~Itrain,:)';

% Build autoencoder
hiddenSize = ??; % <-- EXPERIMENT WITH DIFFERENT COMPRESSION RATIOS!
autoenc = trainAutoencoder(Xtrain,hiddenSize,...
        'EncoderTransferFunction','logsig',...
        'DecoderTransferFunction','satlin',...
        'L2WeightRegularization',0.005,...  <-- Try to change!
        'SparsityRegularization',1,...      <-- Try to change!
        'SparsityProportion',.05);          <-- Try to change!

figure('name','Traing data')
for i = 1:20
    subplot(4,5,i);
    PlotZip(Xtrain(:,i))
end

% Calculate reconstructed training data
Xpred = predict(autoenc,Xtrain);
MSEtrain = mean((Xpred(:)-Xtrain(:)).^2);

figure('name','Traing data, reconstructed')
for i = 1:20
    subplot(4,5,i);
    PlotZip(Xpred(:,i))
end

% Code and decode test data
Xcomp   = encode(autoenc,Xtest);
Xdecode = decode(autoenc,Xcomp);
MSEtest = mean((Xdecode(:)-Xtest(:)).^2);

figure('name','Test data')
for i = 1:20
    subplot(4,5,i);
    PlotZip(Xtest(:,i))
end

figure('name','Test data, reconstructed')
for i = 1:20
    subplot(4,5,i);
    PlotZip(Xdecode(:,i))
end

w = autoenc.DecoderWeights;
figure('name','Decoder weights')
for i = 1:min(36,hiddenSize)
    subplot(6,6,i);
    PlotZip(w(:,i))
end

w = autoenc.EncoderWeights;
figure('name','Encoder weights')
for i = 1:min(36,hiddenSize)
    subplot(6,6,i);
    PlotZip(w(i,:))
end

fprintf('MSE training = %f, MSE test = %f\n',MSEtrain,MSEtest);

end


function PlotZip(X)
    c=reshape(X,16,16);
    c=rot90(c,1);
    c = [c;zeros(1,16)]; %pcolor does not show last row and column
    c = [c zeros(17,1)];
    pcolor(-c)
    caxis([0 .5])
    set(gca,'xtick',[],'ytick',[])
    axis square
end % PlotZip %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

