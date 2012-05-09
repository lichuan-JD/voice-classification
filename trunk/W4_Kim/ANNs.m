% Artificial Neural Networks

clear
%load('H:\Kurser_undervisning\TINONS1\Uge4\classification_2D_trainset')
load('classification_2D_trainset')
scatter(x1, y1, 'r'), hold on, scatter(x2,y2, 'b')

% Using netlab..
%cd('H:\Kurser_undervisning\TINONS1\Tools\netlab\netlab')
% Set up network parameters.
nin = 2;                % Number of inputs.
nhidden = 10;			% Number of hidden units.
nout = 2;               % Number of outputs.
outputfunc = 'logistic';  % output function
alpha = 0.01;			% Coefficient of weight-decay prior. 

% create network (object)
net = mlp(nin, nhidden, nout, outputfunc, alpha);

% Set up vector of options for the optimiser.
options = zeros(1,18);
options(1) = 1;			% This provides display of error values.
options(14) = 100;		% Number of training cycles. 

% Train using scaled conjugate gradients.
[net, options] = netopt(net, options, train_data, labels_t, 'scg');

% Test on test set
%load('H:\Kurser_undervisning\TINONS1\Uge4\classification_2D_testset')
load('classification_2D_testset')
scatter(x1, y1, 200, 'r'), hold on, scatter(x2,y2, 200, 'b')
y_est = mlpfwd(net, test_data);
[max_val,max_id] = max(y_est'); % find max. values
t_est = max_id - 1 ; % id is 1,2,3.. in matlab - not 0,1,2..
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


% calculate test and training set errors
%load('H:\Kurser_undervisning\TINONS1\Uge4\classification_2D_trainset')
load('classification_2D_trainset')
y_est = mlpfwd(net, train_data);
Ctrain = confmat(y_est, labels_t);
err_train = 1-sum(diag(Ctrain))/sum(Ctrain(:)) % correct classification percentage


% on large test set..
%load('H:\Kurser_undervisning\TINONS1\Uge4\large_classification_2D_testset')
load('large_classification_2D_testset')
y_est = mlpfwd(net, test_data);
Ctest = confmat(y_est, labels_t)
err_test = 1-sum(diag(Ctest))/sum(Ctest(:)) % correct classification percentage

