clear;
close all;
load fisheriris;

%kernel = 'rbf'; % Gaussian Radial Basis Funciton kernel
%kernel = 'quadratic'; % Quadratic kernel
%kernel = 'polynomial'; % Polynomial kernel with order of 3
kernel = 'mlp'; % Multilayer Perceptron kernel with diefault scale and bias
%kernel = 'linear'; % Linear

data = [meas(:,1), meas(:,2)];
groups = ismember(species,'setosa');
%groups = ismember(species,'versicolor');
%groups = ismember(species,'viriginica');
[train, test] = crossvalind('holdOut',groups);
cp = classperf(groups);
figure
svmStruct = svmtrain(data(train,:),groups(train), 'Kernel_Function', kernel, 'showplot',true);

title(sprintf('Kernel Function: %s',...
               func2str(svmStruct.KernelFunction)), 'interpreter','none');
pause(3);
classes = svmclassify(svmStruct,data(test,:),'showplot',true);
classperf(cp,classes,test);
error = 1 - cp.CorrectRate

%figure
%svmStruct = svmtrain(data(train,:),groups(train),...
%                     'showplot',true,'boxconstraint',1e6);          
%classes = svmclassify(svmStruct,data(test,:),'showplot',true);   
%classperf(cp,classes,test);
%cp.CorrectRate