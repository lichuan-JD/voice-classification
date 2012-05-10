function [Ctrain, Ctest] = ANN3D(Ynew, Ytnew, Wnew, Wtnew, Znew, dimensions)

N1 = size(Ynew,1);
train_data = [Ynew ; Wnew ; Znew];
test_data = [Ytnew ; Wtnew ; Znew];
labels_t(:,1) = [ones(N1,1) ; zeros(N1,1) ; zeros(N1, 1)];
labels_t(:,2) = [zeros(N1,1) ; ones(N1,1) ; zeros(N1, 1)];
labels_t(:,3) = [zeros(N1,1) ; zeros(N1,1) ; ones(N1, 1)];

x1 = Ynew(:,1);
y1 = Ynew(:,2);
z1 = Ynew(:,3);
x2 = Wnew(:,1);
y2 = Wnew(:,2);
z2 = Wnew(:,3);
x3 = Znew(:,1);
y3 = Znew(:,2);
z3 = Znew(:,3);

figure(2)
scatter3(x1, y1, z1, 'r'), hold on, scatter3(x2, y2, z2, 'g'), scatter3(x3, y3, z3,'b');

%% Set up network parameters.
% Artificial Neural Networks
outputfunc = 'logistic';  % output function
%outputfunc = 'softmax';  % output function
nin = dimensions;                % Number of inputs.
nout = 3;               % Number of outputs.

% Parameters to vary
nhidden = 5;			% Number of hidden units.
alpha = 0.25;			% Coefficient of weight-decay prior. 

% Set up vector of options for the optimiser.
options = zeros(1,18);
options(1) = 1;			% This provides display of error values.
opt = 50;		% Number of training cycles. 

% Find best hiden units
%start = 1;
%stop = 30;
%for nhidden = start:stop 

% Find best training cycles
%start = 10;
%stop = 100;
%for opt = start:stop

% Find best alpha
start = 0.01;
stop = 0.05;
idx = 1;
for alpha = start:0.01:stop    
    options(14) = opt;
    % create network (object)
    net = mlp(nin, nhidden, nout, outputfunc, alpha);
    % Train using scaled conjugate gradients.
    [net, options] = netopt(net, options, train_data, labels_t, 'scg');

    % Test on training set
    y_est = mlpfwd(net, train_data);
    Ctrain = confmat(y_est, labels_t);
    err_train(idx) = 1-sum(diag(Ctrain))/sum(Ctrain(:)) % correct classification percentage
   
    % Test on test set
    %load('classification_2D_testset')
    % 
    y_est = mlpfwd(net, test_data);
    Ctest = confmat(y_est, labels_t);
    err_test(idx) = 1-sum(diag(Ctest))/sum(Ctest(:)) % correct classification percentage

    idx = idx + 1;
end;

x = start:0.01:stop;
%x = start:stop;
figure(3), 
hold on
plot(x, err_train, 'b');
plot(x, err_test, 'r');
title('train error(blue) vs. test error (red)');
xlabel('alpha');

% calculate test and training set errors
y_est = mlpfwd(net, train_data);
Ctrain = confmat(y_est, labels_t)
err_train = 1-sum(diag(Ctrain))/sum(Ctrain(:)) % correct classification percentage

% on large test set..
y_est = mlpfwd(net, test_data);
Ctest = confmat(y_est, labels_t)
err_test = 1-sum(diag(Ctest))/sum(Ctest(:)) % correct classification percentage


