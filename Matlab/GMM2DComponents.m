function [Ctrain, Ctest] = GMM2DComponents(Ynew, Ytnew, Wnew, Wtnew, ncentres)

figure, scatter(Ynew(:,1), Ynew(:,2), 'r'), hold on,
scatter(Wnew(:,1), Wnew(:,2), 'b')
title('GMM2D training set Voice 1 (red) Voice2 (blue) used for validation');

%% Voice 2
data = Ynew(:,[1 2]);

% plot data
xi=min(data(:,1)); xf=max(data(:,1)); 
yi=min(data(:,2)); yf=max(data(:,2));

dimensions = 2;
covartype = 'diag'; % covariance-matrix type.. 'spherical', 'diag' or 'full'
mix = gmm(dimensions, ncentres, covartype);

opts = foptions; % standard options
%opts(1) = 1; % show errors
opts(1) = 0; % don't show errors
opts(3) = 0.0001; % stop-criterion of EM-algorithm
opts(5) = 1; % do not reset covariance matrix in case of small singular values.. (1=do reset..)
opts(14) = 100; % max number of iterations
%[MIX, OPTIONS, ERRLOG] = GMMEM(MIX, X, OPTIONS)
[mix, opts, errlog] = gmmem(mix, data, opts);

mixV1 = mix % see contents..

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
title('Gaussian Mixture for Voice1');

%% Voice 2
data = Wnew(:,[1 2]);

% plot data
xi=min(data(:,1)); xf=max(data(:,1)); 
yi=min(data(:,2)); yf=max(data(:,2));

dimensions = 2;
covartype = 'diag'; % covariance-matrix type.. 'spherical', 'diag' or 'full'
mix = gmm(dimensions, ncentres, covartype);

opts = foptions; % standard options
%opts(1) = 1; % show errors
opts(1) = 0; % don't show errors
opts(3) = 0.0001; % stop-criterion of EM-algorithm
opts(5) = 1; % do not reset covariance matrix in case of small singular values.. (1=do reset..)
opts(14) = 100; % max number of iterations
%[MIX, OPTIONS, ERRLOG] = GMMEM(MIX, X, OPTIONS)
[mix, opts, errlog] = gmmem(mix, data, opts);

mixV2 = mix % see contents..

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
title('Gaussian Mixture for Voice2');

%% 2D example - classification against test set 

d = ncentres; % Number of classes

% Selecting test sets
test_v1 = Ytnew;
test_v2 = Wtnew;

tM1 = size(test_v1, 1); % Number of test samples
tM2 = size(test_v2, 1); 

figure, scatter(test_v1(:,1), test_v1(:,2), 'r'), hold on,
scatter(test_v2(:,1), test_v2(:,2), 'b')
title('GMM2D test set Voice 1 (red) Voice2 (blue) used for validation');

for k = 1:ncentres

    % 4. Posterior class probabilities P(C | x) using Bayes’ theorem
    % Prior probabiliies
    % 5. Now, we’re ready to use P(C | x) on a new (test) set. 
    covarV1 = zeros([dimensions dimensions]);
    covarV2 = zeros([dimensions dimensions]);
    for i = 1:dimensions
        covarV1(i,i) = mixV1.covars(k,i);
        covarV2(i,i) = mixV2.covars(k,i);
    end
    pC1_GMM1(k,:) = normalDensity(d, test_v1, mixV1.centres(k,:), covarV1)*mixV1.priors(k);
    pC1_GMM2(k,:) = normalDensity(d, test_v1, mixV2.centres(k,:), covarV2)*mixV2.priors(k);
    pC2_GMM1(k,:) = normalDensity(d, test_v2, mixV1.centres(k,:), covarV1)*mixV1.priors(k);
    pC2_GMM2(k,:) = normalDensity(d, test_v2, mixV2.centres(k,:), covarV2)*mixV2.priors(k);
end

for k = 1:ncentres
    pC1_GMM1(k,:) = pC1_GMM1(k,:) / sum(pC1_GMM1(k,:));
    pC1_GMM2(k,:) = pC1_GMM2(k,:) / sum(pC1_GMM2(k,:));
    pC2_GMM1(k,:) = pC2_GMM1(k,:) / sum(pC2_GMM1(k,:));
    pC2_GMM2(k,:) = pC2_GMM2(k,:) / sum(pC2_GMM2(k,:));
end

% Product of all Gausian Mixtures log-likelihoods - projection to each class 1, 2
for i=1:tM1
    pC_v1(1, i) = 1;
    pC_v1(2, i) = 1;
    for k = 1:ncentres
        pC_v1(1, i) = pC_v1(1, i) * pC1_GMM1(k,i);
        pC_v1(2, i) = pC_v1(2, i) * pC1_GMM2(k,i);
    end
end    
for i=1:tM2
    pC_v2(1, i) = 1;
    pC_v2(2, i) = 1;
    for k = 1:ncentres
        pC_v2(1, i) = pC_v2(1, i) * pC2_GMM1(k,i);
        pC_v2(2, i) = pC_v2(2, i) * pC2_GMM2(k,i);
    end
end



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

x1 = test_v1(:,1);
y1 = test_v1(:,2);
scatter(x1(t_est(1:tM1)==1), y1(t_est(1:tM1)==1), 'bx'),
x2 = test_v2(:,1);
y2 = test_v2(:,2);
scatter(x2(t_est(tM1+1:end)==0), y2(t_est(tM1+1:end)==0), 'rx');

Ctest = confmat(t, t_est') % uses PRTools
err_test = 1-sum(diag(Ctest))/sum(Ctest(:)) % correct classification percentage


Ctrain = 0;

figure,
hold on
plot(log(pC_v1(1,:)), 'r');
plot(log(pC_v1(2,:)), 'b');
title('GMM2D test if V1 belongs to GMV1(red), GMV2(blue)');

figure,
hold on
plot(log(pC_v2(1,:)), 'r');
plot(log(pC_v2(2,:)), 'b');
title('GMM2D test if V2 belongs to GMV1(red), GMV2(blue)');

