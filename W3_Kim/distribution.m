function [x, py] = distribution(vector, step)

x = floor(min(vector)):step:ceil(max(vector));
hy = hist(vector, x);
py = hy/sum(hy);