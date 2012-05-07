function [t_est, W] = linear2Dboundary(class1,  class2)

x1 = class1(:, 1); % Red
y1 = class1(:, 2);

x2 = class2(:, 1); % Green
y2 = class2(:, 2);

M1 = size(x1, 1); % Number of samples from class 1
M2 = M1; % Number of samples from class 2
%M1 = 20;
%M2 = 20;
figure, scatter(x1(1:M1), y1(1:M1), 'r'), hold on, scatter(x2(1:M2),y2(1:M2), 'g')
title('Class 1 - red, Class 2 - green');
t(:,1) = [ones(M1,1) ; zeros(M2,1)];
t(:,2) = [zeros(M1,1) ; ones(M2,1)];

figure
hold on
C1 = [[x1(1:M1)] [y1(1:M1)]];
scatter(C1(:,1)', C1(:,2)', 'r');
MC1 = mean(C1)
scatter(MC1(1), MC1(2), 'k');
C2 = [[x2(1:M2)] [y2(1:M2)]];
MC2 = mean(C2)
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

% Adjust samples with distance > 7 to mean values
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
Ctest = confmat(t(:,2), t_est') % as expected, lower error on training set than test set..
err_test = 1-sum(diag(Ctest))/sum(Ctest(:)) % correct classification percentage

scatter(x1(t_est(1:M1)==1), y1(t_est(1:M1)==1), 'bx')
scatter(x2(t_est(M1+1:end)==0), y2(t_est(M1+1:end)==0), 'kx')
% decision boundary
dwx = W(1,1)-W(1,2); dwy = W(2,1)-W(2,2); dwbias = W(3,1)-W(3,2);
%xdecbound = linspace(5,20,30); % simply plotpoints, PCA
xdecbound = linspace(-2,2,30); % simply plotpoints, MDA
%xdecbound = linspace(-6,10,30); % simply plotpoints, MDA
plot(xdecbound, -(dwx/dwy)*xdecbound - (dwbias/dwy), 'k') 
xlabel('e1');
ylabel('e2');
title('Decision Boundary (Class 1 - red, Class 2 - green)');

% Histograms for distance to mean center of class 1 and 2
%figure, hist(R1, 20);
%title('Class 1 - red');
%figure, hist(R2, 20);
%title('Class 2 - green');
