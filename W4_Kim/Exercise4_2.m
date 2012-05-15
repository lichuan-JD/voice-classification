% Artificial Neural Networks
% Experiment with the “ANNs.m”-file and the 2D, 2-class model.
% Try to vary the different parameters in a systematic way - e.g. make a plot of the train/test
% classification errors as function of number of hidden units.
% What is the meaning of the outputfunctions - „linear?, „logistic? and „softmax? ?
%   Related to activation function
% What is the form of the error/cost function ?
%
% Which optimisation method do we use ?

close all;
clear

load('classification_2D_trainset')
figure(1)
scatter(x1, y1, 'r'), hold on, scatter(x2,y2, 'b')

% Set up network parameters.
outputfunc = 'logistic';  % output function
%outputfunc = 'softmax';  % output function
nin = 2;                % Number of inputs.
nout = 2;               % Number of outputs.

% Parameters to vary
nhidden = 5;			% Number of hidden units.
alpha = 0.3;			% Coefficient of weight-decay prior. 

% Set up vector of options for the optimiser.
options = zeros(1,18);
options(1) = 1;			% This provides display of error values.
opt = 30;		% Number of training cycles. 

idx = 1;

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
stop = 2.0;
for alpha = start:0.01:stop    
    options(14) = opt;
    load('classification_2D_trainset')
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
    load('large_classification_2D_testset')
    y_est = mlpfwd(net, test_data);
    Ctest = confmat(y_est, labels_t);
    err_test(idx) = 1-sum(diag(Ctest))/sum(Ctest(:)) % correct classification percentage

    idx = idx + 1;
end;

x = start:0.01:stop;
%x = start:stop;
figure(2), 
hold on
plot(x, err_train, 'b');
plot(x, err_test, 'r');
title('train error(blue) vs. test error (red)');
xlabel('alpha');

%% ANN.m reference 
% Test on test set
load('classification_2D_testset')
figure(1)
scatter(x1, y1, 200, 'r'), hold on, scatter(x2,y2, 200, 'b')
y_est = mlpfwd(net, test_data);
[max_val,max_id] = max(y_est'); % find max. values
t_est = max_id - 1 ; % id is 1,2,3.. in matlab - not 0,1,2..
figure(1)
scatter(x1(t_est(1:N1)==1), y1(t_est(1:N1)==1), 200, 'bx')
scatter(x2(t_est(N1+1:end)==0), y2(t_est(N1+1:end)==0), 200, 'rx')

% draw decision boundary
xi=-2; xf=4; yi=-2; yf=4; 
inc=0.01;
xrange = xi:inc:xf;
yrange = yi:inc:yf;
[X Y]=meshgrid(xrange, yrange);
ygrid = mlpfwd(net, [X(:) Y(:)]);
ygrid(:,1) = ygrid(:,1)-ygrid(:,2); % difference between output 1 and 2 - will be 0 at decision boundary
ygrid = reshape(ygrid(:,1),size(X));
figure(1), contour(xrange, yrange, ygrid, [0 0], 'k-')


% % calculate test and training set errors
% load('classification_2D_trainset')
% y_est = mlpfwd(net, train_data);
% Ctrain = confmat(y_est, labels_t);
% err_train = 1-sum(diag(Ctrain))/sum(Ctrain(:)) % correct classification percentage
% 
% 
% % on large test set..
% load('large_classification_2D_testset')
% y_est = mlpfwd(net, test_data);
% Ctest = confmat(y_est, labels_t)
% err_test = 1-sum(diag(Ctest))/sum(Ctest(:)) % correct classification percentage

