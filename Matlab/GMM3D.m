function [Ctrain, Ctest] = GMM3D(Ynew, Ytnew, Wnew, Wtnew, Znew, Ztnew, dimensions)

data = [Ynew; Wnew; Znew];

ncentres = 3; % number of mixtures - try using e.g. 3, 5 and 7..
covartype = 'diag'; % covariance-matrix type.. 'spherical', 'diag' or 'full'
mix = gmm(dimensions, ncentres, covartype);

opts = foptions; % standard options
%opts(1) = 1; % show errors
opts(1) = 0; % don't show errors
opts(3) = 0.001; % stop-criterion of EM-algorithm
opts(5) = 0; % do not reset covariance matrix in case of small singular values.. (1=do reset..)
opts(14) = 50; % max number of iterations
[mix, opts, errlog] = gmmem(mix, data, opts);

mix % see contents..

% classification against test set 
% Selecting test sets
tstC1 = [Ynew; Ytnew];
tstC2 = [Wnew; Wtnew];

tM1 = size(tstC1, 1); % Number of test samples
tM2 = size(tstC2, 1);

figure, scatter(tstC1(:,1), tstC1(:,2), 'r'), hold on, scatter(tstC2(:,1),tstC2(:,2), 'g')
title('GMM3D train+test used for validation (2 dimensions)');

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
title('GMM3D test if V1 belongs to GM1(red), GM2(blue), GM3(green)');

figure, 
hold on
plot(pC_v2(1,:), 'r');
plot(pC_v2(2,:), 'g');
plot(pC_v2(3,:), 'b');
title('GMM3D test if V2 belongs to GM1(red), GM2(blue), GM3(green)');

k = 1;
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

