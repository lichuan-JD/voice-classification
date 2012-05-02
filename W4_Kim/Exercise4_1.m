% Artificial Neural Networks

close all
clear
x = linspace(-5, 5, 500);
s = sinc(x);
t = sinc(x) + 0.05*randn(1,500);
figure, plot(x,t);

% Using netlab..
% Set up network parameters.
nin = 1;                % Number of inputs.
%nhidden = 10;			% Number of hidden units.
nhidden = 20;			% Number of hidden units.
nout = 1;               % Number of outputs.
outputfunc = 'linear';  % output function
%alpha = 0.5;			% Coefficient of weight-decay prior. 
alpha = 0.001;			% Coefficient of weight-decay prior. 

% create network (object)
net = mlp(nin, nhidden, nout, outputfunc, alpha);

% Set up vector of options for the optimiser.
options = zeros(1,18);
options(1) = 1;			% This provides display of error values.
%options(14) = 200;		% Number of training cycles. 
options(14) = 400;		% Number of training cycles. 

% Train using scaled conjugate gradients.
[net, options] = netopt(net, options, x', t', 'scg');

y_est = mlpfwd(net, x');
figure, 
hold on, plot(y_est, 'b'), plot(s, 'r');
