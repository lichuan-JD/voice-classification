
clear
close all
% dimensionality reduction - by PCA
im = imread('face1.jpg');
% im = imread('colors1.jpg');  
imshow(im)

X = double(reshape(im, [size(im,1)*size(im,2) 3]));
whos

% Plotting in 3D
Nplot = 10000; % number of points to plot.. 
pointId = randperm(size(im,1)*size(im,2)); % make random numbers - to get "representative samples"
pointId = pointId(1:Nplot); % choose some of them..
figure, scatter3(X(pointId, 1), X(pointId, 2), X(pointId, 3), 'r.')

% PCA - Principal Component Analysis
N = size(X,1);
mu = mean(X);
sigma = (1/N)*(X - repmat(mu, N, 1))'*(X - repmat(mu, N, 1));      % calculate the covariance matrix estimate
[v,d] = eig(sigma)  % find eigenvectors and eigenvalues
d = diag(d); % keep only non-zero entries..
d/sum(d)    % percentages

% we choose only to use the dominant directions.. with largest
% eigenvalues
v1 = v(:,[1 3]); % between 2 and 3 largest..
%v1 = v(:,3); % between 3 largest..
Y = (X-repmat(mu, N, 1))*v1; % new 2D representation / feature vectors
hist(Y, 100)


% check quality after "decompression/decoding".. since v1*v1' = I, since
% orthonormal basisvectors
imvec_rec = Y*v1' + repmat(mu, N, 1);
im_rec = uint8(reshape(imvec_rec, size(im)));
figure, imshow(im_rec)








