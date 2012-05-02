
close all
clear

%% MNIST handwritten digits data
load('mnist_all.mat')

% show some data..
%for i=1:10,
%    im = reshape(train5(i,:), 28, 28)';
%    figure(1), imshow(im),
%    pause
%end

X = double(train2);  % definition of our dataset
im = reshape(X(10,:), 28, 28)';
figure(1), imshow(im) % show image 10

% Compute mean image
N = size(X,1);
mu = mean(X);
figure(2), imshow(uint8(reshape(mu, 28, 28))') % plot the mean picture

% PCA - Principal Component Analysis
sigma = (1/N)*(X - repmat(mu, N, 1))'*(X - repmat(mu, N, 1));      % calculate the covariance matrix estimate
[v,d] = eig(sigma);  % find eigenvectors and eigenvalues
d = diag(d); % keep only non-zero entries..

% PCA - Alternative method usign SVD
%[u, s, v] = svd(X - repmat(mu, N, 1));  % find singular values decomposition..
%d = (1/N)*diag(s).^2; % singular values..
%figure, plot(d/sum(d)); % plots dimensions
%figure, plot(cumsum(d/sum(d)));

% we choose only to use the dominant directions.. with largest
% singular values.. in this case we uses M
%M = 10; % Number of dominant directions (number of features or eigenvectors)
M = 5;
colormap(gray);
figure(3), imagesc(reshape(v(:, 1), 28, 28)')    % plot the largest eigenvector..
v1= v(:, [1 M]); % using five features from the 700 dimentions
Xnew = X*v1; % projecting onto the new basis

% Restore image
%Y = (X(10,:)-repmat(mu, 1, 1))*v1; % new 2D representation / feature vectors
Y = (X-repmat(mu, N, 1))*v1; % new 2D representation / using feature vectors
figure(4), hist(Y, 100) % Plotting histogram
% check quality after "decompression/decoding".. since v1*v1' = I, since
% orthonormal basisvectors
imvec_rec = Y*v1' + repmat(mu, N, 1); % decompression of all images
im_rec = uint8(reshape(imvec_rec(10,:), size(im))); % Select image 10
figure(5), imshow(im_rec') % Show decompressed image 10

% test on other class.. test3
Z = double(test3);
figure(6), imshow(reshape(Z(2,:), 28, 28)')
Znew = Z*v1; % projection on the same basis..

% show new data in 2D feature space..
figure(7)
scatter(Xnew(:,1), Xnew(:,2), 'r.'), hold on
scatter(Znew(:,1), Znew(:,2), 'b.')

