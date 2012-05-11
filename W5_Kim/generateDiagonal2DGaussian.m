function [y, u1] = generateDiagonal2DGaussian(N, mean, direction)

% Example of generating Gaussian data in 2D
x = randn(N,2); % generate white data - ie. normally distributed, covariance matrix = I (identity), zero-mean

u1 = direction; % direction of one semi-axis
mu = mean; % mean value

% generate data
u1 = u1/norm(u1); % make it unit vector
u2 = [u1(2) -u1(1)]; % orthonormal vector to u1
U = [u1' u2']; % transformation matrix
D = [u1(1) 0; 0 u1(2)]; % eigenvalue matrix

y = x*D*U + repmat(mean, N, 1);
