function [t_est, Ctest] = gausianDiscriminant(Ynew, Ytnew, Wnew, Wtnew)

M1 = size(Ynew, 1); % Number of samples from class 1
M2 = size(Wnew, 1); % Number of samples from class 2

tC1 = Ynew;
tC2 = Wnew;

figure, scatter(tC1(:,1), tC1(:,2), 'r'), hold on, scatter(tC2(:,1),tC2(:,2), 'g')
title('Baysian Training set');

% 1. + 2. + 3. Compute mean and covariance used in normal distribution
meanC1 = mean(tC1)
meanC2 = mean(tC2)

% calculate the covariance matrix estimate
%sigmaC1 = cov(tC1)
%sigmaC2 = cov(tC2)
sigmaC1 = (1/M1)*(tC1 - repmat(meanC1, M1, 1))'*(tC1 - repmat(meanC1, M1, 1))      
sigmaC2 = (1/M2)*(tC2 - repmat(meanC2, M1, 1))'*(tC2 - repmat(meanC2, M2, 1))      

%% 2D example - classification against test set 

d = 2; % Number of classes
tM1 = size(Ytnew, 1); % Number of test samples
tM2 = size(Wtnew, 1);

% Selecting test sets
tstC1 = Ytnew;
tstC2 = Wtnew;

figure, scatter(tstC1(:,1), tstC1(:,2), 'r'), hold on, scatter(tstC2(:,1),tstC2(:,2), 'g')
title('Baysian Test set');

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
plot(pC1_x, 'r');
plot(pC2_x, 'b');
title('Baysian Test if V1 belongs to C1(red) or C2(blue)');
 
%pC1_y = normalDensity(d, tstC2, meanC1, sigmaC1)*PC2;
%pC2_y = normalDensity(d, tstC2, meanC2, sigmaC2)*PC2;
pC1_y = bayesLogDiscriminator(tstC2, meanC1, sigmaC1, PC2);
pC2_y = bayesLogDiscriminator(tstC2, meanC2, sigmaC2, PC2);

figure, 
hold on
plot(pC1_y, 'r');
plot(pC2_y, 'b');
title('Baysian Test if V2 belongs to C1(red) or C2(blue)');

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

Ctest = confmat(t, t_est') % uses PRTools
err_test = 1-sum(diag(Ctest))/sum(Ctest(:)) % correct classification percentage

