clear
close all;

% Example of generating Gaussian data in 2D
N = 1000;

% generate white data - ie. normally distributed, 
% covariance matrix = I (identity), zero-mean
x = randn(N,2); 

u1 = [5 1]; % direction of one semi-axis
d1 = 5; % lenghts of semi-axes 1
d2 = 1;
mu = [0 0]; % mean value

% generate data
u1 = u1/norm(u1); % make it unit vector
u2 = [u1(2) -u1(1)]; % orthonormal vector to u1
U = [u1' u2']; % transformation matrix
D = [d1 0; 0 d2]; % eigenvalue matrix

y = x*D*U + repmat(mu, N, 1);

scatter(y(:,1), y(:,2)), grid on
g=line([mu(1) d1*u1(1)+mu(1)], [mu(2) d1*u1(2)+mu(2)])
set(g, 'Color', [0 0 0], 'LineWidth', 3)

% Histograms of the two dimensions independently - they look gaussian
% They should look gausian since - 
% generated white data (normally distributed)
[x, py] = distribution(y(:,1), 1);
figure, plot(x, py, ':s'); 
title('P(x1)');
[x, py] = distribution(y(:,2), 0.5);
figure, plot(x, py, ':s'); 
title('P(x2)');

% Estimate the conditional distribution P(x2 | x1 > 5) - ie. the probability 
% distribution of x2 given that x1 is larger than 5. In Matlab, this can be 
% done by first finding the indexes where x1 is larger than 5
% (eg. “id = y(: , 1) > 5”), then copying corresponding values of x2 into a 
% new variable (eg. “ynew = y(id, 2)”) and finally finding histogram of ynew.

id = find(y(:,1) > 5);
ynew = y(id,:);

[x, py] = distribution(ynew(:,2), 0.5);
figure, plot(x, py, ':s'); 
title('P(x2|x1>5)');

%Estimate the covariance matrix and the mean vector from the data - 
%do they correspond to the expected values (the values used to generate the data..) ?
mean_y = mean(y)
covariance_y = diag(cov(y, 0)) % Covariance normalized by N (1)
mean_ynew = mean(ynew)
covariance_ynew = diag(cov(ynew), 0)% Covariance normalized by N (1)



