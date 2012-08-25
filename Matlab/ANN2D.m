function [Ctrain, Ctest] = ANN2D(Ynew, Ytnew, Wnew, Wtnew, Znew, Ztnew, outputs)

% Create train and test data set for ANN 2D
N1 = size(Ynew,1);
Nt1 = size(Ytnew,1);

if outputs == 3 % Number of output classes
    labels_t(:,1) = [ones(N1,1) ; zeros(N1,1) ; zeros(N1, 1)];
    labels_t(:,2) = [zeros(N1,1) ; ones(N1,1) ; zeros(N1, 1)];
    labels_t(:,3) = [zeros(N1,1) ; zeros(N1,1) ; ones(N1, 1)];  
    labels_tt(:,1) = [ones(Nt1,1) ; zeros(Nt1,1) ; zeros(Nt1, 1)];
    labels_tt(:,2) = [zeros(Nt1,1) ; ones(Nt1,1) ; zeros(Nt1, 1)];
    labels_tt(:,3) = [zeros(Nt1,1) ; zeros(Nt1,1) ; ones(Nt1, 1)];  
    train_data = [Ynew(:,[1 2]) ; Wnew(:,[1 2]) ; Znew(:,[1 2])];
    test_data = [Ytnew(:,[1 2]) ; Wtnew(:,[1 2]) ; Ztnew(:,[1 2])];
else
    labels_t(:,1) = [ones(N1,1) ; zeros(N1,1)];
    labels_t(:,2) = [zeros(N1,1) ; ones(N1,1)];
    labels_tt(:,1) = [ones(Nt1,1) ; zeros(Nt1,1)];
    labels_tt(:,2) = [zeros(Nt1,1) ; ones(Nt1,1)];
    train_data = [Ynew(:,[1 2]) ; Wnew(:,[1 2])];
    test_data = [Ytnew(:,[1 2]) ; Wtnew(:,[1 2])];
end

x1 = Ynew(:,1);
y1 = Ynew(:,2);
x2 = Wnew(:,1);
y2 = Wnew(:,2);
x3 = Znew(:,1);
y3 = Znew(:,2);

xt1 = Ytnew(:,1);
yt1 = Ytnew(:,2);
xt2 = Wtnew(:,1);
yt2 = Wtnew(:,2);
xt3 = Znew(:,1);
yt3 = Znew(:,2);


%% Set up network parameters.
% Artificial Neural Networks
outputfunc = 'logistic';  % output function
%outputfunc = 'softmax';  % output function
nin = 2;                % Number of inputs.
nout = outputs;         % Number of outputs.

% Parameters to vary
nhidden = 8;			% Number of hidden units.
alpha = 0.001;			% Coefficient of weight-decay prior. 

% Set up vector of options for the optimiser.
options = zeros(1,18);
%options(1) = 1;			% This provides display of error values.
options(1) = 0;			
opt = 50;		% Number of training cycles. 

idx = 1;

% Find best hiden units
%start = 1;
%stop = 30;
%for nhidden = start:stop 

% Find best training cycles
%start = 5;
%stop = 100;
%for opt = start:stop

% Find best alpha
%start = 0.1;
%stop = 0.2;
%for alpha = start:0.01:stop 

start = 40;
stop = 50;
for opt = start:stop
    options(14) = opt;
    % create network (object)
    net = mlp(nin, nhidden, nout, outputfunc, alpha);
    % Train using scaled conjugate gradients.
    [net, options] = netopt(net, options, train_data, labels_t, 'scg');

    % Test on training set
    y_est = mlpfwd(net, train_data);
    Ctrain = confmat(y_est, labels_t);
    err_trainv(idx) = 1-sum(diag(Ctrain))/sum(Ctrain(:)); % correct classification percentage
   
    % Test on test set
    y_est = mlpfwd(net, test_data);
    Ctest = confmat(y_est, labels_tt);
    err_testv(idx) = 1-sum(diag(Ctest))/sum(Ctest(:)); % correct classification percentage

    idx = idx + 1;
end;

%x = start:0.01:stop;
x = start:stop;
figure, 
hold on
plot(x, err_trainv, 'b');
plot(x, err_testv, 'r');
title('ANN2D train error(blue) vs. test error (red)');
xlabel('training iterations');

% Plot training set
figure,
scatter(x1, y1, 'r'), hold on, scatter(x2, y2, 'g'), 
if outputs == 3 % Number of output classes
    scatter(x3, y3, 'b');
end
y_est = mlpfwd(net, train_data);
[max_val,max_id] = max(y_est'); % find max. values
t_est = max_id - 1 ; % id is 1,2,3.. in matlab - not 0,1,2..
if outputs == 2 % Number of output classes
    scatter(x1(t_est(1:N1)==1), y1(t_est(1:N1)==1), 200, 'gx')
    scatter(x2(t_est(N1+1:end)==0), y2(t_est(N1+1:end)==0), 200, 'rx')
end

% draw decision boundary
xi=-10; xf=10; yi=-8; yf=8; 
%xi=-4; xf=2; yi=-3; yf=2; 
inc=0.01;
xrange = xi:inc:xf;
yrange = yi:inc:yf;
[X Y]=meshgrid(xrange, yrange);
ygrid = mlpfwd(net, [X(:) Y(:)]);
ygrid(:,1) = ygrid(:,1)-ygrid(:,2); % difference between output 1 and 2 - will be 0 at decision boundary
ygrid = reshape(ygrid(:,1),size(X));
contour(xrange, yrange, ygrid, [0 0], 'k-')
title('ANN2D countors on train set');


% Plot test set
figure,
scatter(xt1, yt1, 'r'), hold on, scatter(xt2, yt2, 'g'), 
if outputs == 3 % Number of output classes
    scatter(xt3, yt3, 'b');
end
y_est = mlpfwd(net, test_data);
[max_val,max_id] = max(y_est'); % find max. values
t_est = max_id - 1 ; % id is 1,2,3.. in matlab - not 0,1,2..
if outputs == 2 % Number of output classes
    scatter(xt1(t_est(1:Nt1)==1), yt1(t_est(1:Nt1)==1), 200, 'gx')
    scatter(xt2(t_est(Nt1+1:end)==0), yt2(t_est(Nt1+1:end)==0), 200, 'rx')
end

% draw decision boundary
%xi=-10; xf=10; yi=-8; yf=8; 
inc=0.01;
xrange = xi:inc:xf;
yrange = yi:inc:yf;
[X Y]=meshgrid(xrange, yrange);
ygrid = mlpfwd(net, [X(:) Y(:)]);
ygrid(:,1) = ygrid(:,1)-ygrid(:,2); % difference between output 1 and 2 - will be 0 at decision boundary
ygrid = reshape(ygrid(:,1),size(X));
contour(xrange, yrange, ygrid, [0 0], 'k-')
title('ANN2D countors on test set');


% calculate test and training set errors
y_est = mlpfwd(net, train_data);
Ctrain = confmat(y_est, labels_t)
err_train = 1-sum(diag(Ctrain))/sum(Ctrain(:)) % correct classification percentage

% on large test set..
y_est = mlpfwd(net, test_data);
Ctest = confmat(y_est, labels_tt)
err_test = 1-sum(diag(Ctest))/sum(Ctest(:)) % correct classification percentage

