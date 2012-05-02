clear

% Example of generating Gaussian data in 2D
N = 1000;
x = randn(N,2); % generate white data - ie. normally distributed, covariance matrix = I (identity), zero-mean

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
