close all
clear

% Apply the Expectation-Maximization (EM) algorithm for GMMs to the artificial data set from
% exercise 1.

% 2D example - classification
load classification_2D_dataset
figure, scatter(x1, y1, 'r'), hold on, scatter(x2,y2, 'b')
t(:,1) = [ones(N1,1) ; zeros(N2,1)];
t(:,2) = [zeros(N1,1) ; ones(N2,1)];
Z = [[x1;x2] [y1;y2] ones(N1+N2,1)];
W = inv(Z'*Z)*Z'*t
% Finding estimate and convert to 0/1
y_est = Z*W;
[max_val,max_id] = max(y_est'); % find max. values
t_est = max_id - 1 ; % id is 1,2,3.. in matlab - not 0,1,2..
% finding testing set error / confusion matrix
confmat(t(:,2), t_est') % as expected, lower error on training set than test set..

scatter(x1(t_est(1:N1)==1), y1(t_est(1:N1)==1), 'bx')
scatter(x2(t_est(N1+1:end)==0), y2(t_est(N1+1:end)==0), 'rx')
% decision boundary
dwx = W(1,1)-W(1,2); dwy = W(2,1)-W(2,2); dwbias = W(3,1)-W(3,2);
xdecbound = linspace(-2,3,30); % simply plotpoints
plot(xdecbound, -(dwx/dwy)*xdecbound - (dwbias/dwy), 'k') 


% 2D example - classification against test set 
%clear
load classification_2D_dataset_large
figure, scatter(x1, y1, 'r'), hold on, scatter(x2,y2, 'b')
t_t(:,1) = [ones(N1,1) ; zeros(N2,1)];
t_t(:,2) = [zeros(N1,1) ; ones(N2,1)];
Z = [[x1;x2] [y1;y2] ones(N1+N2,1)];
%W = inv(Z'*Z)*Z'*t
y_est_t = Z*W;
[max_val_t,max_id_t] = max(y_est_t'); % find max. values
t_est_t = max_id_t - 1 ; % id is 1,2,3.. in matlab - not 0,1,2..
confmat(t_t(:,2), t_est_t') % as expected, lower error on training set than test set..

scatter(x1(t_est_t(1:N1)==1), y1(t_est_t(1:N1)==1), 'bx')
scatter(x2(t_est_t(N1+1:end)==0), y2(t_est_t(N1+1:end)==0), 'rx')
% decision boundary
dwx = W(1,1)-W(1,2); dwy = W(2,1)-W(2,2); dwbias = W(3,1)-W(3,2);
xdecbound = linspace(-2,3,30); % simply plotpoints
plot(xdecbound, -(dwx/dwy)*xdecbound - (dwbias/dwy), 'k') 


%%  This is the “training/learning” phase where the parameters of the model are estimated
% (inferred). The file “GMMs.m” can be used as a starting point.
% Try first with 2 mixtures and see if the correct parameters are found. Next try with a larger number
% of mixtures - e.g. 10 mixtures. Do you get any problems ?

load classification_2D_dataset_large
x = [x1; x2];
y = [y1; y2];
data = [x y];

% plot data
xi=-3; xf=6; yi=-3; yf=6;
figure, scatter(x, y)
axis([xi xf yi yf])

% Using netlab..
%cd('H:\Kurser_undervisning\TINONS1\Tools\netlab\netlab')

dim = 2; % dimensions
%ncentres = 2; % number of mixtures - try using e.g. 3, 5 and 7..
ncentres = 7; % number of mixtures - try using e.g. 3, 5 and 7..
covartype = 'diag'; % covariance-matrix type.. 'spherical', 'diag' or 'full'
mix = gmm(dim, ncentres, covartype);

opts = foptions; % standard options
opts(1) = 1; % show errors
opts(3) = 0.001; % stop-criterion of EM-algorithm
opts(5) = 0; % do not reset covariance matrix in case of small singular values.. (1=do reset..)
opts(14) = 50; % max number of iterations
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

