
clear
%cd('H:\Kurser_undervisning\TINONS1\Uge5')
load cluster_dataset
data = [x y];

% plot data
xi=-3; xf=5; yi=-5; yf=6;
scatter(x, y)
axis([xi xf yi yf])

% Using netlab..
%cd('H:\Kurser_undervisning\TINONS1\Tools\netlab\netlab')

dim = 2; % dimensions
ncentres = 7; % number of mixtures - try using e.g. 3, 5 and 7..
covartype = 'spherical'; % covariance-matrix type.. 'spherical', 'diag' or 'full'
mix = gmm(dim, ncentres, covartype);


opts = foptions; % standard options
opts(1) = 1; % show errors
opts(3) = 0.001; % stop-criterion of EM-algorithm
opts(5) = 0; % do not reset covariance matrix in case of small singular values.. (1=do reset..)
opts(14) = 100; % max number of iterations
%[MIX, OPTIONS, ERRLOG] = GMMEM(MIX, X, OPTIONS)
[mix, opts, errlog] = gmmem(mix, data, opts);

mix % see contents..

% draw contours..
inc=0.01;
xrange = xi:inc:xf;
yrange = yi:inc:yf;
[X Y]=meshgrid(xrange, yrange);
ygrid = gmmprob(mix, [X(:) Y(:)]);
ygrid = reshape(ygrid,size(X));
figure, imagesc(ygrid(end:-1:1, :)), colorbar
figure, contour(xrange, yrange, ygrid, 0:0.01:0.3,'k-')
hold on, scatter(x, y, 'b')



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% step-by-step illustration of EM algorithm for GMMs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M = 500; % number of steps/iterations
mix = gmm(dim, ncentres, covartype); % new mixture
%mix = gmminit(mix, data, opts); % initialize using Kmeans
opts(14) = 1; % max number of iterations
for iterId=1:M, 
    [mix, opts, errlog] = gmmem(mix, data, opts);

    % draw contours..
    figure(3), scatter(x, y, 'r'), hold on
    a = gmmactiv(mix, [X(:) Y(:)]); % activations
    for i=1:ncentres,
        ygrid = a(:,i); % gaussian mixture number i
        ygrid = reshape(ygrid,size(X));
        contour(xrange, yrange, ygrid, 0:0.1:1,'k-')
        plot(mix.centres(i, 1), mix.centres(i,2), 'k.', 'MarkerSize', 10)
        title(['Iteration ' num2str(iterId) ' - Error ' num2str(errlog)], 'FontSize', 20)
    end
    axis([xi xf yi yf])
    hold off
    pause
end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GMM for handwritten digits (unsupervised learning)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear

%% MNIST handwritten digits data
%load('H:\Kurser_undervisning\TINONS1\DataSets\MNIST_HandwrittenDigits\mnist_all.mat')
load('mnist_all.mat')

% show some data..
for i=1:5,
    im = reshape(test0(i,:), 28, 28)';
    figure(1), imshow(im),
    pause
end

% use 0's, 1's and 2's (random..)
X = double([test0; test1; test2]);

mu = mean(X);
[u, s, v] = svd(X - repmat(mu, size(X,1), 1));

dim = 10; % dimensions
v1= v(:, 1:dim); % using new basis/features
Xnew = X*v1; 

% dim = 784; % dimensions
ncentres = 7; % number of mixtures
covartype = 'diag'; % covariance-matrix type.. 'spherical', 'diag' or 'full'
mix = gmm(dim, ncentres, covartype);

opts = foptions; % standard options
opts(1) = 1; % show errors
opts(3) = 0.001; % stop-criterion of EM-algorithm
opts(5) = 1; % do not reset covariance matrix in case of small singular values.. (1=do reset..)
opts(14) = 100; % max number of iterations

mix = gmminit(mix, Xnew, opts); % initialize using Kmeans
[mix, opts, errlog] = gmmem(mix, Xnew, opts);

% generate data (deterministic from the means..) - an advantage of
% generative models..
for i=1:ncentres,
    center_im = mix.centres(i,:)*v1' + mu; % mix.centres(1,:) is "how much is there of each principal component=basis vector"
    im = reshape(center_im, 28, 28)';
    figure(1), imagesc(im),colorbar % note - also negative values..
    pause
end





