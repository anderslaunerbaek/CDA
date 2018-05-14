
load zipdata
Y = y;

dim = ??;  % <-- TRY DIFFERENT DIMENSIONS

% Create Self Organizing Network
net = selforgmap([dim dim],100,3,'gridtop');

% Train the Network
[net,tr] = train(net,X');

% Test the Network
outputs = net(X')';

% View the Network
view(net)

[~, bin] = max(outputs,[],2);

[x, y] = meshgrid([1:dim],[1:dim]);
xpos = reshape(x,1,dim*dim);
ypos = reshape(y,1,dim*dim);

figure
for i=1:length(bin)
   x = xpos(bin(i)) + .6*rand(1)-.3;
   y = ypos(bin(i)) + .6*rand(1)-.3;
   text(x,y,num2str(Y(i)));
   hold on
end
axis([0 dim+1 0 dim+1])
box on

