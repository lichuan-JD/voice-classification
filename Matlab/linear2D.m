function [Ctrain, Ctest, W] = linear2D(train1, test1, train2, test2)

x1 = train1(:, 1); % Red
y1 = train1(:, 2);

x2 = train2(:, 1); % Green
y2 = train2(:, 2);

M1 = size(x1, 1); % Number of samples from class 1
M2 = M1; % Number of samples from class 2
t(:,1) = [ones(M1,1) ; zeros(M2,1)];
t(:,2) = [zeros(M1,1) ; ones(M2,1)];

C1 = [[x1(1:M1)] [y1(1:M1)]];
MC1 = mean(C1)
C2 = [[x2(1:M2)] [y2(1:M2)]];
MC2 = mean(C2)

% Compute mean (center) of samples for class 1 and 2 and distance to center
for r=1:size(C1,1)
    T1(r) = (C1(r,1) - MC1(1))^2 + (C1(r,2) - MC1(2))^2;
end
R1 = sqrt(T1);

for r=1:size(C2,1)
    T2(r) = (C2(r,1) - MC2(1))^2 + (C2(r,2) - MC2(2))^2;
end
R2 = sqrt(T2);

% Adjust samples with distance > 7 to mean values
%R_MAX = 2; 
R_MAX = 7; 
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
Ctrain = confmat(t(:,2), t_est') % as expected, lower error on training set than test set..
err_train = 1-sum(diag(Ctrain))/sum(Ctrain(:)) % correct classification percentage

%% Draw training result
figure, 
scatter(x1(1:M1), y1(1:M1), 'r'), 
hold on, 
scatter(x2(1:M2),y2(1:M2), 'g'),
scatter(x1(t_est(1:M1)==1), y1(t_est(1:M1)==1), 'bx'),
scatter(x2(t_est(M1+1:end)==0), y2(t_est(M1+1:end)==0), 'kx');
scatter(MC1(1), MC1(2), 'kd');
scatter(MC2(1), MC2(2), 'kd');

% Draw decision boundary
dwx = W(1,1)-W(1,2); dwy = W(2,1)-W(2,2); dwbias = W(3,1)-W(3,2);
xmin = min([min(x1(1:M1)) min(x2(1:M2))]);
xmax = max([max(x1(1:M1)) max(x2(1:M2))]);
xdecbound = linspace(xmin,xmax,30);
plot(xdecbound, -(dwx/dwy)*xdecbound - (dwbias/dwy), 'k') 
xlabel('e1');
ylabel('e2');
title('Training linear2D (Class 1 - red, Class 2 - green)');

% Histograms for distance to mean center of class 1 and 2
%figure, hist(R1, 20);
%title('Class 1 - red');
%figure, hist(R2, 20);
%title('Class 2 - green');

%% Test against test classes
xt1 = test1(:, 1); % Red
yt1 = test1(:, 2);

xt2 = test2(:, 1); % Green
yt2 = test2(:, 2);

M1 = size(xt1, 1); % Number of samples from class 1
M2 = M1; % Number of samples from class 2
Zt = [[xt1(1:M1);xt2(1:M2)] [yt1(1:M1);yt2(1:M2)] ones(M1+M2,1)];
yt_est = Zt*W; % Use same weights as for training

[max_tval,max_tid] = max(yt_est'); % find max. values
tt_est = max_tid - 1 ; % id is 1,2,3.. in matlab - not 0,1,2..
% finding testing set error / confusion matrix
Ctest = confmat(t(:,2), tt_est') % as expected, lower error on training set than test set..
err_test = 1-sum(diag(Ctest))/sum(Ctest(:)) % correct classification percentage

% Draw test result
figure, 
scatter(xt1(1:M1), yt1(1:M1), 'r'), 
hold on, 
scatter(xt2(1:M2),yt2(1:M2), 'g');
dwx = W(1,1)-W(1,2); dwy = W(2,1)-W(2,2); dwbias = W(3,1)-W(3,2);
xmin = min([min(xt1(1:M1)) min(xt2(1:M2))]);
xmax = max([max(xt1(1:M1)) max(xt2(1:M2))]);
xdecbound = linspace(xmin,xmax,30);
plot(xdecbound, -(dwx/dwy)*xdecbound - (dwbias/dwy), 'k');
xlabel('e1');
ylabel('e2');
title('Test linear2D (Class 1 - red, Class 2 - green)');


