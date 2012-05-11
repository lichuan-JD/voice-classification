close all
clear

%Create an artificial data set that is drawn from a Gaussian Mixture Model (GMM) in 2 dimensions,
%that is, a probability distribution p(x) = ?p(k) p(x|k) where p(x|k) is a gaussian distribution. As a
%starting point, “generate_Gaussian_data.m” from week 3 can be used. Use eg. K = 2 mixtures with
%mixture coefficients p(1) = ?1 = 0.3 and p(2) = ?2 = 0.7. Choose the mean vectors for the two
%mixtures as [0 0] and [4 4], respectively. The covariance matrices are 1*I and 2*I, respectively,
%where I is the unity matrix ([1 0; 0 1]). Create e.g. N=1000 samples. Although not strictly correct,
%you can simply create 300 samples from mixture 1 and 700 from mixture 2, to achieve (roughly)
%correct mixture coefficients.

% Example of generating Gaussian data in 2D
N1 = 300;
N2 = 700;
d1 = 1; % Diagonal of covariance matrix
d2 = 2;
mu1 = [0 0]; % mean value
mu2 = [4 4];
pk1 = 0.3;
pk2 = 0.7;

[pxk1, u1] = generateDiagonal2DGaussian(N1, mu1, [d1 d1]);
[pxk2, u2] = generateDiagonal2DGaussian(N2, mu2, [d2 d2]);

figure
scatter(pxk1(:,1), pxk1(:,2), 'b.'), grid on, hold on
g=line([mu1(1) d1*u1(1)+mu1(1)], [mu1(2) d1*u1(2)+mu1(2)])
set(g, 'Color', [0 0 0], 'LineWidth', 1)

scatter(pxk2(:,1), pxk2(:,2), 'g.')
g=line([mu2(1) d2*u2(1)+mu2(1)], [mu2(2) d2*u2(2)+mu2(2)])
set(g, 'Color', [0 0 0], 'LineWidth', 1)


px = [pk1*pxk1; pk2*pxk2]
% Plot the samples of p(x) = ?p(k) p(x|k) where p(x|k) is a gaussian distribution
figure
scatter(px(:,1), px(:,2), 'b.'), grid on, hold on

%Calculate the probability p(x = [2 0]), that is, the value of the probability distribution function at the
%point x=[2 0]. Does the result correspond to the plot of the samples ?

x = [2 0];
px1 = probabilityDiagonal2DGaussian(x, mu1, [d1 d1])*pk1;
px2 = probabilityDiagonal2DGaussian(x, mu2, [d2 d2])*pk2;
p = px1+px2
scatter(x(1), x(2), 'r')
% The probability for px1 = 0.0065 and px2 = 0.000375 meaning that the
% probability is very small p = 0.0068

%Calculate the so-called responsibilities for the point x=[2 0] from the two mixtures, that is, calculate
%p(k=1 | x=[2 0]) and p(k=2 | x=[2 0]). Do similarly for the point x=[6 0] and explain the result from
%the knowledge of the GMM parameters (means, covariances and mixture priors).
px1 = probabilityDiagonal2DGaussian([2 0], mu1, [d1 d1])*pk1
px2 = probabilityDiagonal2DGaussian([2 0], mu2, [d2 d2])*pk2
px1 = probabilityDiagonal2DGaussian([6 0], mu1, [d1 d1])*pk1
px2 = probabilityDiagonal2DGaussian([6 0], mu2, [d2 d2])*pk2

% GMM on artificial data set
dim = 2; % dimensions
%ncentres = 3; % number of mixtures - try using e.g. 3, 5 and 7..
ncentres = 2; % number of mixtures - try using e.g. 3, 5 and 7..
covartype = 'diag'; % covariance-matrix type.. 'spherical', 'diag' or 'full'
mix = gmm(dim, ncentres, covartype);

opts = foptions; % standard options
opts(1) = 1; % show errors
opts(3) = 0.001; % stop-criterion of EM-algorithm
opts(5) = 0; % do not reset covariance matrix in case of small singular values.. (1=do reset..)
opts(14) = 100; % max number of iterations
%[MIX, OPTIONS, ERRLOG] = GMMEM(MIX, X, OPTIONS)
[mix, opts, errlog] = gmmem(mix, px, opts);

mix % see contents..

% draw contours..
xi=-1; xf=5; yi=-1; yf=5;
inc=0.01;
xrange = xi:inc:xf;
yrange = yi:inc:yf;
[X Y]=meshgrid(xrange, yrange);
ygrid = gmmprob(mix, [X(:) Y(:)]);
ygrid = reshape(ygrid,size(X));
figure, imagesc(ygrid(end:-1:1, :)), colorbar
figure, contour(xrange, yrange, ygrid, 0:0.01:0.3,'k-')
hold on, scatter(x, y, 'b')

