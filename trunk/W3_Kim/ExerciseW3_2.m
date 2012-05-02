clear
close all
% Experiment with the linear classifier using “intro.m”. Try on the dataset
% “classification_2D_dataset_large”.
% Train the classifier on a training set, test it on a test set and find classification error/accuracy and
% confusion matrix for both train and test set.
%1. Choose probabilistic model/distribution for p(x|C)
%   e.g. Gaussian distribution
%   (Sometimes) choose initial model parameters
%2. Number of mixtures, hyperparameters,..
%3. Infer/learn parameters of the model by maximizing the Log-likelihood function for each class separately – using a training data set
%    e.g. µ and covariance for the Gaussian for each class
%4. Find the posterior class probabilities P(C | x) using Bayes’ theorem (see next slide for details..)
%5. Now, we’re ready to use P(C | x) on a new (test) set. 


%% 2D example - classification training set (part of large dataset)
load classification_2D_dataset_large

M1 = 100; % Number of samples from class 1
M2 = 100; % Number of samples from class 2

tC1 = [x1(1:M1)  y1(1:M1)];
tC2 = [x2(1:M2)  y2(1:M2)];

figure, scatter(tC1(:,1), tC1(:,2), 'r'), hold on, scatter(tC2(:,1),tC2(:,2), 'b')
title('Training set');

% 1. + 2. + 3. Compute mean and covariance used in normal distribution
meanC1 = mean(tC1);
meanC2 = mean(tC2);
covarC1 = cov(tC1)
covarC2 = cov(tC2)
% calculate the covariance matrix estimate
sigmaC1 = (1/M1)*(tC1 - repmat(meanC1, M1, 1))'*(tC1 - repmat(meanC1, M1, 1))      
sigmaC2 = (1/M2)*(tC2 - repmat(meanC2, M1, 1))'*(tC2 - repmat(meanC2, M2, 1))      

%% 2D example - classification against test set 

d = 2; % Number of classes
N = 9; 
tM1 = N*M1; % Number of tests 
tM2 = N*M2;

% Selecting test sets
tstC1 = [x1(M1+1:(N+1)*M1)  y1(M1+1:(N+1)*M1)];
tstC2 = [x2(M2+1:(N+1)*M2)  y2(M2+1:(N+1)*M2)];

figure, scatter(tstC1(:,1), tstC1(:,2), 'r'), hold on, scatter(tstC2(:,1),tstC2(:,2), 'b')
title('Test set');

% 4. Posterior class probabilities P(C | x) using Bayes’ theorem
% Prior probabiliies
PC1 = M1/(M1+M2);
PC2 = M2/(M1+M2);

% 5. Now, we’re ready to use P(C | x) on a new (test) set. 
%pC1_x = normalDensity(d, tstC1, meanC1, sigmaC1)*PC1;
%pC2_x = normalDensity(d, tstC1, meanC2, sigmaC2)*PC1;
pC1_x = bayesLogDiscriminator(tstC1, meanC1, sigmaC1, PC1);
pC2_x = bayesLogDiscriminator(tstC1, meanC2, sigmaC2, PC1);

figure, 
hold on
plot(pC1_x, 'b');
plot(pC2_x, 'r');
title('Test if x belongs to C1(blue) or C2(red)');
 
%pC1_y = normalDensity(d, tstC2, meanC1, sigmaC1)*PC2;
%pC2_y = normalDensity(d, tstC2, meanC2, sigmaC2)*PC2;
pC1_y = bayesLogDiscriminator(tstC2, meanC1, sigmaC1, PC2);
pC2_y = bayesLogDiscriminator(tstC2, meanC2, sigmaC2, PC2);

figure, 
hold on
plot(pC1_y, 'r');
plot(pC2_y, 'b');
title('Test if y belongs to C1(red) or C2(blue)');

% Confusion matrix validation
t = [zeros(tM1,1) ; ones(tM2,1)];
% Test for x (C1) belongs to C1 or C2
for i=1:tM1
    if pC1_x(i) > pC2_x(i)
        t_est(i) = 0;
    else
        t_est(i) = 1;
    end;
end;

% Test for y (C2) belongs to C1 or C2
for i=1:tM2
    if pC1_y(i) > pC2_y(i)
        t_est(i+tM1) = 0;
    else
        t_est(i+tM1) = 1;
    end;
end;
confmat(t, t_est') % uses PRTools


