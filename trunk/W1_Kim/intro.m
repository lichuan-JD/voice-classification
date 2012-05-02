clear
close all

% linear regression
load regression_dataset
Z = [ones(N,1) x'];
W = inv(Z'*Z)*Z'*y'  % w = (XT*X)-1 XT * T
plot(x,y, 'r.'), hold on
plot(x, W(1) + W(2)*x, 'b')

% polynomial (linear) regression
Z = [ones(N,1) x' x'.^2 x'.^3 x'.^4 x'.^5 x'.^6 x'.^7];
W = inv(Z'*Z)*Z'*y'
plot(x,y, 'r.'), hold on
plot(x, W(1) + W(2)*x + W(3)*x.^2 + W(4)*x.^3 + W(5)*x.^4 + W(6)*x.^5 + W(7)*x.^6 + W(8)*x.^7, 'b')

% polynomial (linear) regression
Z = [ones(N,1) x' x'.^2 x'.^3];
W = inv(Z'*Z)*Z'*y'
plot(x,y, 'r.'), hold on
plot(x, W(1) + W(2)*x + W(3)*x.^2 + W(4)*x.^3, 'g')


clear
% classification - 1-of-K coding.. training phase
load classification_trainset
t(:,1) = [ones(N1,1) ; zeros(N2,1)];
t(:,2) = [zeros(N1,1) ; ones(N2,1)];
Z = [x ones(N1+N2,1)];
W = inv(Z'*Z)*Z'*t
figure, plot(x1, zeros(N1,1), 'ro'), hold on, plot(x2, zeros(N2,1), 'bx')
axis([-3 6 -.1 1.1])
plot(x, W(1,1)*x + W(2,1), 'r')
plot(x, W(1,2)*x + W(2,2), 'b')

% finding training set error / confusion matrix
y_est = W'*[x1 x2; ones(1,N1+N2)];
[max_val,max_id] = max(y_est); % find max. values
t_est = max_id - 1 ;
confmat([t1 t2]', t_est') % uses PRTools

% classification - testing
load classification_testset
plot(x1, zeros(N1,1), 'ro', 'MarkerSize', 20)
plot(x2, zeros(N2,1), 'bx', 'MarkerSize', 20)

% finding testing set error / confusion matrix
y_est = W'*[x1 x2; ones(1,N1+N2)]
[max_val,max_id] = max(y_est); % find max. values
t_est = max_id - 1 ; % id is 1,2,3.. in matlab - not 0,1,2..
confmat(t, t_est') % as expected, lower error on training set than test set..

 
%%% Problems with the linear discriminant..
% eg. outliers 
clear
load classification_trainset
t(:,1) = [ones(N1,1) ; zeros(N2,1)];
t(:,2) = [zeros(N1,1) ; ones(N2,1)];
Z = [x ones(N1+N2,1)];
x_outlier = -12;
x_outlier2 = -10;
Z = [Z; x_outlier 1; x_outlier2 1];    % add outlier
t = [t; 0 1; 0 1];       % to class 2
W = inv(Z'*Z)*Z'*t
figure, 
plot(x1, zeros(N1,1), 'ro'), hold on, plot(x2, zeros(N2,1), 'bx')
plot(x_outlier, 0, 'kd')
plot(x_outlier2, 0, 'kd')
axis([-15 6 -.1 1.1])
plot(x, W(1,1)*x + W(2,1), 'r') % observe - decision boundary moved a lot..
plot(x, W(1,2)*x + W(2,2), 'b')



% 2D example - classification
clear
load classification_2D_dataset
figure, scatter(x1, y1, 'r'), hold on, scatter(x2,y2, 'b')
t(:,1) = [ones(N1,1) ; zeros(N2,1)];
t(:,2) = [zeros(N1,1) ; ones(N2,1)];
Z = [[x1;x2] [y1;y2] ones(N1+N2,1)];
W = inv(Z'*Z)*Z'*t
y_est = Z*W;
[max_val,max_id] = max(y_est'); % find max. values
t_est = max_id - 1 ; % id is 1,2,3.. in matlab - not 0,1,2..
scatter(x1(t_est(1:N1)==1), y1(t_est(1:N1)==1), 'bx')
scatter(x2(t_est(N1+1:end)==0), y2(t_est(N1+1:end)==0), 'rx')
% decision boundary
dwx = W(1,1)-W(1,2); dwy = W(2,1)-W(2,2); dwbias = W(3,1)-W(3,2);
xdecbound = linspace(-2,3,30); % simply plotpoints
plot(xdecbound, -(dwx/dwy)*xdecbound - (dwbias/dwy), 'k') 

% 2D example - classification
clear
load classification_2D_dataset_large
figure, scatter(x1, y1, 'r'), hold on, scatter(x2,y2, 'b')
t(:,1) = [ones(N1,1) ; zeros(N2,1)];
t(:,2) = [zeros(N1,1) ; ones(N2,1)];
Z = [[x1;x2] [y1;y2] ones(N1+N2,1)];
W = inv(Z'*Z)*Z'*t
y_est = Z*W;
[max_val,max_id] = max(y_est'); % find max. values
t_est = max_id - 1 ; % id is 1,2,3.. in matlab - not 0,1,2..
scatter(x1(t_est(1:N1)==1), y1(t_est(1:N1)==1), 'bx')
scatter(x2(t_est(N1+1:end)==0), y2(t_est(N1+1:end)==0), 'rx')
% decision boundary
dwx = W(1,1)-W(1,2); dwy = W(2,1)-W(2,2); dwbias = W(3,1)-W(3,2);
xdecbound = linspace(-2,3,30); % simply plotpoints
plot(xdecbound, -(dwx/dwy)*xdecbound - (dwbias/dwy), 'k') 
