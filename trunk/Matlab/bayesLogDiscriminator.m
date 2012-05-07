function [pX] = bayesLogDiscriminator(x, mean, covar, px)
% Computes the discriminat functions as described 
% in equation (69) page 41, Richard O. Duda, Pattern Classification
% (Where the covariance matrices are different for each category)

%pX = zeros(size(x,1));
for r=1:size(x,1)
  z1 = (x(r,:) - mean);
  z2 = (z1 / covar) * z1';
  z3 = log(det(covar));
  pX(r) = -0.5*z2 - 0.5*z3 + log(px);
end