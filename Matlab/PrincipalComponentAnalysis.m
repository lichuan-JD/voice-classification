function [principalComponents] = PrincipalComponentAnalysis(featureSet, subSet)

N = size(featureSet,1);
mu = mean(featureSet);

% PCA - Principal Component Analysis
% calculate the covariance matrix estimate
sigma = (1/N)*(featureSet - repmat(mu, N, 1))'*(featureSet - repmat(mu, N, 1));      
[v,d] = eig(sigma);  % find eigenvectors and eigenvalues
d = diag(d); % keep only non-zero entries..

figure,
plot(d);
title('PCA eigenvalues');


% we choose only to use the dominant directions.. with largest singular values..
principalComponents= v(:, subSet); % using subSet of principal components from projection

