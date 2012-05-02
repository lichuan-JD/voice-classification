
close all
clear

%% MNIST handwritten digits data
load('mnist_all.mat')

% show some data..
for i=1:10,
    im = reshape(train5(i,:), 28, 28)';
    figure(1), imshow(im),
    pause
end

X = double(train5);  % definition of our dataset
figure(1), imshow(reshape(X(10,:), 28, 28)') % image 10
% PCA - Principal Component Analysis
N = size(X,1);
mu = mean(X);
figure, imshow(uint8(reshape(mu, 28, 28))')    % plot the mean..

[u, s, v] = svd(X - repmat(mu, N, 1));  % find singular values decomposition..
d = (1/N)*diag(s).^2; % singular values..
figure, plot(d/sum(d)); % plots dimensions

figure, plot(cumsum(d/sum(d)));

% we choose only to use the dominant directions.. with largest
% singular values..
imagesc(reshape(v(:, 1), 28, 28)')    % plot the largest eigenvector..
v1= v(:, 1:2); % using two features from the 700 dimentions
Xnew = X*v1; % projecting onto the new basis


% test on other class..
Z = double(test1);
figure, imshow(reshape(Z(1,:), 28, 28)')
Znew = Z*v1; % projection on the same basis..

% show new data in 2D feature space..
scatter(Xnew(:,1), Xnew(:,2), 'r.'), hold on
scatter(Znew(:,1), Znew(:,2), 'b.')






