function [Ctrain, Ctest] = GMM2D(Ynew, Ytnew, Wnew, Wtnew, Znew, Ztnew)

data = [Ynew(:,[1 2]); Wnew(:,[1 2]); Znew(:,[1 2])];

% plot data
xi=min(data(:,1)); xf=max(data(:,1)); 
yi=min(data(:,2)); yf=max(data(:,2));
figure, scatter(Ynew(:,1), Ynew(:,2), 'r'), hold on
scatter(Wnew(:,1), Wnew(:,2), 'g')
scatter(Znew(:,1), Znew(:,2), 'b')
axis([xi xf yi yf])
title('GMM2D training data V1(red), V2(green), S(blue)');

dimensions = 2;
ncentres = 3; % number of mixtures - try using e.g. 3, 5 and 7..
covartype = 'diag'; % covariance-matrix type.. 'spherical', 'diag' or 'full'
mix = gmm(dimensions, ncentres, covartype);

opts = foptions; % standard options
%opts(1) = 1; % show errors
opts(1) = 0; % don't show errors
opts(3) = 0.001; % stop-criterion of EM-algorithm
opts(5) = 1; % do not reset covariance matrix in case of small singular values.. (1=do reset..)
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
hold on, scatter(data(:,1), data(:,2), 'y')
title('Gaussian Mixture for Voice1, Voice2 and Silence');

%% 2D example - classification against test set 

d = 2; % Number of classes

% Selecting test sets
tstC1 = [Ynew; Ytnew];
tstC2 = [Wnew; Wtnew];

tM1 = size(tstC1, 1); % Number of test samples
tM2 = size(tstC2, 1);

figure, scatter(tstC1(:,1), tstC1(:,2), 'r'), hold on, scatter(tstC2(:,1),tstC2(:,2), 'g')
title('GMM2D train+test used for validation');

for k = 1:mix.ncentres

    % 4. Posterior class probabilities P(C | x) using Bayes’ theorem
    % Prior probabiliies
    % 5. Now, we’re ready to use P(C | x) on a new (test) set. 
    covar = zeros([dimensions dimensions]);
    for i = 1:dimensions
        covar(i,i) = mix.covars(k,i);
    end
    pC_v1(k,:) = bayesLogDiscriminator(tstC1, mix.centres(k,:), covar, mix.priors(k));
    pC_v2(k,:) = bayesLogDiscriminator(tstC2, mix.centres(k,:), covar, mix.priors(k));

end

figure, 
hold on
plot(pC_v1(1,:), 'r');
plot(pC_v1(2,:), 'g');
plot(pC_v1(3,:), 'b');
title('GMM2D test if V1 belongs to GM1(red), GM2(blue), GM3(green)');

figure, 
hold on
plot(pC_v2(1,:), 'r');
plot(pC_v2(2,:), 'g');
plot(pC_v2(3,:), 'b');
title('GMM2D test if V2 belongs to GM1(red), GM2(blue), GM3(green)');

k = 2;
% Confusion matrix validation k = 1 and k = 2
t = [zeros(tM1,1) ; ones(tM2,1)];
% Test for x (C1) belongs to C1 or C2
for i=1:tM1
    if pC_v1(k, i) > pC_v1(k+1, i)
        t_est(i) = 1;
    else
        t_est(i) = 0;
    end;
end;

% Test for y (C2) belongs to C1 or C2
for i=1:tM2
    if pC_v2(k, i) > pC_v2(k+1, i)
        t_est(i+tM1) = 1;
    else
        t_est(i+tM1) = 0;
    end;
end;

Ctest = confmat(t, t_est') % uses PRTools
err_test = 1-sum(diag(Ctest))/sum(Ctest(:)) % correct classification percentage

Ctrain = 0;

