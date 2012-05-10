function [Ctrain, Ctest, W] = linear3D(train1, train2, test1, test2)

x1 = train1(:, 1); % Red
y1 = train1(:, 2);
z1 = train1(:, 3);

x2 = train2(:, 1); % Green
y2 = train2(:, 2);
z2 = train2(:, 3);

M1 = size(x1, 1); % Number of samples from class 1
M2 = M1; % Number of samples from class 2

t(:,1) = [ones(M1,1) ; zeros(M2,1)];
t(:,2) = [zeros(M1,1) ; ones(M2,1)];

Z = [[x1(1:M1);x2(1:M2)] [y1(1:M1);y2(1:M2)] [z1(1:M1);z2(1:M2)] ones(M1+M2,1)];
W = inv(Z'*Z)*Z'*t

% Finding estimate and convert to 0/1
y_est = Z*W;

% finding testing set error / confusion matrix
[max_val,max_id] = max(y_est'); % find max. values
t_est = max_id - 1 ; % id is 1,2,3.. in matlab - not 0,1,2..
Ctrain = confmat(t(:,2), t_est') % as expected, lower error on training set than test set..
err_train = 1-sum(diag(Ctrain))/sum(Ctrain(:)) % correct classification percentage

%% Plots classes in training feature space
figure
C1 = [x1(1:M1) y1(1:M1) z1(1:M1)];
scatter3(C1(:,1)', C1(:,2)', C1(:,3)', 'r.');
hold on;
MC1 = mean(C1); % Class1 mean
scatter3(MC1(1), MC1(2), MC1(3),'r');
C2 = [x2(1:M2) y2(1:M2) z2(1:M2)];
MC2 = mean(C2); % Class2 mean
scatter3(C2(:,1)', C2(:,2)', C2(:,3)','g.');
scatter3(MC2(1), MC2(2), MC2(3), 'g');

scatter3(x1(t_est(1:M1)==1), y1(t_est(1:M1)==1), z1(t_est(1:M1)==1), 'kh')
scatter3(x2(t_est(M1+1:end)==0), y2(t_est(M1+1:end)==0), z1(t_est(M1+1:end)==0), 'kh')

% decision boundary
dwx = W(1,1)-W(1,2); dwy = W(2,1)-W(2,2); dwz = W(3,1)-W(3,2); dwbias = W(4,1)-W(4,2);
%xdecbound = linspace(5,20,30); % simply plotpoints PCA coordinates
xmin = min([min(x1(1:M1)) min(x2(1:M2))]);
xmax = max([max(x1(1:M1)) max(x2(1:M2))]);
xdecbound = linspace(xmin,xmax,30);
plot3(xdecbound, -(dwx/dwz)*xdecbound - (dwbias/dwy), -(dwx/dwz)*xdecbound - (dwbias/dwz), 'k') 
xlabel('e1');
ylabel('e2');
zlabel('e3');
title('Training plan boundary(Class1 - red, Class2 - green)');

%% Test against test classes
xt1 = test1(:, 1); % Red
yt1 = test1(:, 2);
zt1 = test1(:, 3);

xt2 = test2(:, 1); % Green
yt2 = test2(:, 2);
zt2 = test2(:, 3);

M1 = size(xt1, 1); % Number of samples from class 1
M2 = M1; % Number of samples from class 2
Zt = [[xt1(1:M1);xt2(1:M2)] [yt1(1:M1);yt2(1:M2)] [zt1(1:M1);zt2(1:M2)] ones(M1+M2,1)];

yt_est = Zt*W; % Use same weights as for training

[max_tval,max_tid] = max(yt_est'); % find max. values
tt_est = max_tid - 1 ; % id is 1,2,3.. in matlab - not 0,1,2..
% finding testing set error / confusion matrix
Ctest = confmat(t(:,2), tt_est') % as expected, lower error on training set than test set..
err_test = 1-sum(diag(Ctest))/sum(Ctest(:)) % correct classification percentage

%% Plots classes in test feature space
figure
Ct1 = [xt1(1:M1) yt1(1:M1) zt1(1:M1)];
scatter3(Ct1(:,1)', Ct1(:,2)', Ct1(:,3)', 'r.');
hold on;
MCt1 = mean(Ct1); % Class1 mean
scatter3(MCt1(1), MCt1(2), MCt1(3),'r');
Ct2 = [xt2(1:M2) yt2(1:M2) zt2(1:M2)];
MCt2 = mean(Ct2); % Class2 mean
scatter3(Ct2(:,1)', Ct2(:,2)', Ct2(:,3)','g.');
scatter3(MCt2(1), MCt2(2), MCt2(3), 'g');

scatter3(xt1(tt_est(1:M1)==1), yt1(tt_est(1:M1)==1), zt1(tt_est(1:M1)==1), 'kh')
scatter3(xt2(tt_est(M1+1:end)==0), yt2(tt_est(M1+1:end)==0), zt1(tt_est(M1+1:end)==0), 'kh')

% decision boundary
dwx = W(1,1)-W(1,2); dwy = W(2,1)-W(2,2); dwz = W(3,1)-W(3,2); dwbias = W(4,1)-W(4,2);
%xdecbound = linspace(5,20,30); % simply plotpoints PCA coordinates
xmin = min([min(x1(1:M1)) min(x2(1:M2))]);
xmax = max([max(x1(1:M1)) max(x2(1:M2))]);
xdecbound = linspace(xmin,xmax,30);
plot3(xdecbound, -(dwx/dwz)*xdecbound - (dwbias/dwy), -(dwx/dwz)*xdecbound - (dwbias/dwz), 'k') 
xlabel('e1');
ylabel('e2');
zlabel('e3');
title('Test plan boundary(Class1 - red, Class2 - green)');