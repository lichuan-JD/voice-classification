function [t_est, W] = linear3Dboundary(class1,  class2)

x1 = class1(:, 1); % Red
y1 = class1(:, 2);
z1 = class1(:, 3);

x2 = class2(:, 1); % Green
y2 = class2(:, 2);
z2 = class2(:, 3);

M1 = size(x1, 1); % Number of samples from class 1
M2 = M1; % Number of samples from class 2

title('Training set (Class1 - red, Class2 - green)');
t(:,1) = [ones(M1,1) ; zeros(M2,1)];
t(:,2) = [zeros(M1,1) ; ones(M2,1)];

figure
% Plots classes in feature space
C1 = [[x1(1:M1)] [y1(1:M1)] [z1(1:M1)]];
scatter3(C1(:,1)', C1(:,2)', C1(:,3)', 'r.');
hold on;
MC1 = mean(C1) % Class1 mean
scatter3(MC1(1), MC1(2), MC1(3),'r');
C2 = [[x2(1:M2)] [y2(1:M2)] [z2(1:M2)]];
MC2 = mean(C2) % Class2 mean
scatter3(C2(:,1)', C2(:,2)', C2(:,3)','g.');
scatter3(MC2(1), MC2(2), MC2(3), 'g');

Z = [[x1(1:M1);x2(1:M2)] [y1(1:M1);y2(1:M2)] [z1(1:M1);z2(1:M2)] ones(M1+M2,1)];
W = inv(Z'*Z)*Z'*t

% Finding estimate and convert to 0/1
y_est = Z*W;
[max_val,max_id] = max(y_est'); % find max. values
t_est = max_id - 1 ; % id is 1,2,3.. in matlab - not 0,1,2..

% finding testing set error / confusion matrix
confmat(t(:,2), t_est') % as expected, lower error on training set than test set..
scatter3(x1(t_est(1:M1)==1), y1(t_est(1:M1)==1), z1(t_est(1:M1)==1), 'kh')
scatter3(x2(t_est(M1+1:end)==0), y2(t_est(M1+1:end)==0), z1(t_est(M1+1:end)==0), 'kh')

% decision boundary
dwx = W(1,1)-W(1,2); dwy = W(2,1)-W(2,2); dwz = W(3,1)-W(3,2); dwbias = W(4,1)-W(4,2);
%xdecbound = linspace(5,20,30); % simply plotpoints PCA coordinates
xdecbound = linspace(-2,0.5,30); % simply plotpoints MDA coordinates
plot3(xdecbound, -(dwx/dwz)*xdecbound - (dwbias/dwy), -(dwx/dwz)*xdecbound - (dwbias/dwz), 'k') 
xlabel('e1');
ylabel('e2');
zlabel('e3');
title('Decision Boundary (Class1 - red, Class2 - green)');


