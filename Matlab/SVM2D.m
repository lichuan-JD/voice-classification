function [Ctrain, Ctest] = SVM2D(Ynew, Ytnew, Wnew, Wtnew)

%kernel = 'rbf'; % Gaussian Radial Basis Funciton kernel
kernel = 'quadratic'; % Quadratic kernel
%kernel = 'polynomial'; % Polynomial kernel with order of 3
%kernel = 'mlp'; % Multilayer Perceptron kernel with diefault scale and bias
%kernel = 'linear'; % Linear

%data = [Ynew(:,[1 2]); Wnew(:,[1 2])]; % Train data with 2 classes and 2 dimensions
sz = size(Ytnew,1);
data = [Ynew(1:sz,[1 2]); Wnew(1:sz,[1 2])]; % Train data with 2 classes and 2 dimensions

% plot data
% xi=min(data(:,1)); xf=max(data(:,1)); 
% yi=min(data(:,2)); yf=max(data(:,2));
% figure, scatter(Ynew(:,1), Ynew(:,2), 'r'), hold on
% scatter(Wnew(:,1), Wnew(:,2), 'g')
% axis([xi xf yi yf])
% title('SVM2D training data V1(red), V2(green)');

%groups = logical([ones(size(Ynew,1),1); zeros(size(Wnew,1),1)]);
groups = logical([ones(sz,1); zeros(sz,1)]);

%[train, test] = crossvalind('holdOut',groups);
%cp = classperf(groups);

figure
svmStruct = svmtrain(data, groups, 'Kernel_Function', kernel, 'showplot',true);

title(sprintf('Kernel Function: %s',...
               func2str(svmStruct.KernelFunction)), 'interpreter','none');

%% SVM test
% NB - data must be the same size as datatest
datatest = [Ytnew(:,[1 2]); Wtnew(:,[1 2])]; % Test data with 2 classes and 2 dimensions
t = [ones(size(Ytnew,1),1); zeros(size(Wtnew,1),1)];
%groupstest = logical(t);

classes = svmclassify(svmStruct, datatest,'showplot',true);
%classperf(cp, classes, groupstest); 
%error = 1 - cp.CorrectRate

Ctest = confmat(t, classes) % uses PRTools
err_test = 1-sum(diag(Ctest))/sum(Ctest(:)) % correct classification percentage

% Plot test set
% xi=min(datatest(:,1)); xf=max(datatest(:,1)); 
% yi=min(datatest(:,2)); yf=max(datatest(:,2));
% figure, scatter(Ytnew(:,1), Ytnew(:,2), 'r'), hold on
% scatter(Wtnew(:,1), Wtnew(:,2), 'g')
% axis([xi xf yi yf])
% title('SVM2D test data V1(red), V2(green)');

Ctrain = 0;

