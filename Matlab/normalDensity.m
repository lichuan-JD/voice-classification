function [pX] = normalDensity(d, x, mean, covar)
% Computes the multivariate normal density functions as described 
% in equation (38) page 33, Richard O. Duda, Pattern Classification

%pX = zeros(size(x,1));
for r=1:size(x,1)
  z1 = (x(r,:) - mean);
  z2 = (z1 / covar) * z1';
  z3 = power((2*pi), d/2)*sqrt(det(covar));
  pX(r) = 1/z3 * exp(-0.5*z2);
end