function [px] = probabilityDiagonal2DGaussian(x, mean, direction)

% Equation (38) book page 33 
% Dimensions
d = 2;
covar = [direction(1) 0; 0 direction(2)];
z1 = (x - mean);
z2 = (z1 / covar) * z1';
z3 = power(det(covar), 0.5);
z4 = power((2*pi), d/2);
px = 1/(z3*z4) * exp(-0.5*z2);