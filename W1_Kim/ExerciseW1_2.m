clear
close all
% Experiment with the linear classifier using “intro.m”. Try on the dataset
% “classification_2D_dataset_large”.
% Train the classifier on a training set, test it on a test set and find classification error/accuracy and
% confusion matrix for both train and test set.
% Experiment with the number of training samples used - does it affect the performance ?
% With 10 - 20 training samples it seems to be sufficient and doesn't
% change performance with more training set of 100 => 78/2000 errors
% 500 training => 76/2000 errors
% 500 training => 42/2000 errors, with filter radius = 2
% 500 training => 40/2000 errors, with filter radius = 1.5
% 500 training => 38/2000 errors, with filter radius = 1

%% 2D example - classification training set (part of large dataset)
load classification_2D_dataset_large
M1 = 500; % Number of samples from class 1
M2 = 500; % Number of samples from class 2
%M1 = 20;
%M2 = 20;
figure, scatter(x1(1:M1), y1(1:M1), 'r'), hold on, scatter(x2(1:M2),y2(1:M2), 'b')
title('Training set');
t(:,1) = [ones(M1,1) ; zeros(M2,1)];
t(:,2) = [zeros(M1,1) ; ones(M2,1)];

figure
hold on
C1 = [[x1(1:M1)] [y1(1:M1)]];
scatter(C1(:,1)', C1(:,2)', 'c');
MC1 = mean(C1)
%MC1 = median(C1);
scatter(MC1(1), MC1(2), 'k');
C2 = [[x2(1:M2)] [y2(1:M2)]];
MC2 = mean(C2)
%MC2 = median(C2);
scatter(C2(:,1)', C2(:,2)', 'g');
scatter(MC2(1), MC2(2), 'k');

% Compute mean (center) of samples for class 1 and 2 and distance to center
for r=1:size(C1,1)
    T1(r) = (C1(r,1) - MC1(1))^2 + (C1(r,2) - MC1(2))^2;
end
R1 = sqrt(T1);

for r=1:size(C2,1)
    T2(r) = (C2(r,1) - MC2(1))^2 + (C2(r,2) - MC2(2))^2;
end
R2 = sqrt(T2);

% Adjust samples with distance > 3 to mean values
% Improves result from 83 to 79 estimated wrong lables (40 samples)
R_MAX = 2; 
%R_MAX = 1.5; % 74 wrong lables
%R_MAX = 2; % 75 wrong lables
%R_MAX = 2.5; 
%R_MAX = 3; % 79 wrong lables
%R_MAX = 4; % 83 wrong lables
s1 = find(R1 > R_MAX);
x1(s1) = MC1(1);
y1(s1) = MC1(2);
s2 = find(R2 > R_MAX);
x2(s2) = MC2(1);
y2(s2) = MC2(2);

Z = [[x1(1:M1);x2(1:M2)] [y1(1:M1);y2(1:M2)] ones(M1+M2,1)];
W = inv(Z'*Z)*Z'*t
% Finding estimate and convert to 0/1
y_est = Z*W;
[max_val,max_id] = max(y_est'); % find max. values
t_est = max_id - 1 ; % id is 1,2,3.. in matlab - not 0,1,2..
% finding testing set error / confusion matrix
confmat(t(:,2), t_est') % as expected, lower error on training set than test set..
% 5% error with 10 training samples for each 2 classifiers
% 5% error with 20 training samples for each 2 classifiers

scatter(x1(t_est(1:M1)==1), y1(t_est(1:M1)==1), 'bx')
scatter(x2(t_est(M1+1:end)==0), y2(t_est(M1+1:end)==0), 'rx')
% decision boundary
dwx = W(1,1)-W(1,2); dwy = W(2,1)-W(2,2); dwbias = W(3,1)-W(3,2);
xdecbound = linspace(-2,6,30); % simply plotpoints
plot(xdecbound, -(dwx/dwy)*xdecbound - (dwbias/dwy), 'k') 

% Histograms for distance to mean center of class 1 and 2
figure, hist(R1, 20);
figure, hist(R2, 20);

%% 2D example - classification against test set 

figure, scatter(x1, y1, 'r'), hold on, scatter(x2,y2, 'b')
title('Test set');
t_t(:,1) = [ones(N1,1) ; zeros(N2,1)];
t_t(:,2) = [zeros(N1,1) ; ones(N2,1)];
Z = [[x1;x2] [y1;y2] ones(N1+N2,1)];
%W = inv(Z'*Z)*Z'*t
y_est_t = Z*W;
[max_val_t,max_id_t] = max(y_est_t'); % find max. values
t_est_t = max_id_t - 1 ; % id is 1,2,3.. in matlab - not 0,1,2..
confmat(t_t(:,2), t_est_t') % as expected, lower error on training set than test set..
% 5.05% error with 10 training samples for each 2 classifiers
% 4.2% error with 20 training samples for each 2 classifiers

scatter(x1(t_est_t(1:N1)==1), y1(t_est_t(1:N1)==1), 'bx')
scatter(x2(t_est_t(N1+1:end)==0), y2(t_est_t(N1+1:end)==0), 'rx')
% decision boundary
dwx = W(1,1)-W(1,2); dwy = W(2,1)-W(2,2); dwbias = W(3,1)-W(3,2);
xdecbound = linspace(-2,6,30); % simply plotpoints
plot(xdecbound, -(dwx/dwy)*xdecbound - (dwbias/dwy), 'k') 

% Computes length of dW, Euclidean norm ||dW||
dW = [dwx dwy];
LW = sqrt(abs(dW*dW'));
r = y_est/LW; % Equal to g(x)/||dW||
d = sqrt(power(r(:,1),2)+power(r(:,2), 2));
figure, hist(d, 20);

