function [Ctrain, Ctest] = GMM3DComponents(Ynew, Ytnew, Wnew, Wtnew, ncentres, dimensions)

figure, scatter(Ynew(:,1), Ynew(:,2), '.r'), hold on,
scatter(Wnew(:,1), Wnew(:,2), 'b')
title('GMM3D training set Voice 1 (red) Voice2 (blue) used for validation');

%% Voice 2
data = Ynew; %(:,[1 2 3 4]);    
    
% plot data
xi=min(data(:,1)); xf=max(data(:,1)); 
yi=min(data(:,2)); yf=max(data(:,2));

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

%% Voice 1
data = Wnew; %(:,[1 2 3 4]);

% plot data
xi=min(data(:,1)); xf=max(data(:,1)); 
yi=min(data(:,2)); yf=max(data(:,2));

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

%% 2D example - classification against test set 

d = ncentres; % Number of classes

% Selecting test sets
test_v1 = Ytnew;
test_v2 = Wtnew;

tM1 = size(test_v1, 1); % Number of test samples
tM2 = size(test_v2, 1); 

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

for i=1:tM1
%    pC_v1(1, i) = sum(pC1_GMM1(:,i));
%    pC_v1(2, i) = sum(pC1_GMM2(:,i));
    pC_v1(1, i) = max(pC1_GMM1(:,i));
    pC_v1(2, i) = max(pC1_GMM2(:,i));
end    
for i=1:tM2
%    pC_v2(1, i) = sum(pC2_GMM1(:,i));
%    pC_v2(2, i) = sum(pC2_GMM2(:,i));
    pC_v2(1, i) = max(pC2_GMM1(:,i));
    pC_v2(2, i) = max(pC2_GMM2(:,i));
end

k = 1;
% Confusion matrix validation k = 1 and k = 2
t = [zeros(tM1,1) ; ones(tM2,1)];
% Test for x (C1) belongs to C1 or C2
for i=1:tM1
    if pC_v1(k, i) > pC_v1(k+1, i)
        t_est(i) = 0;
    else
        t_est(i) = 1;
    end;
end;

% Test for y (C2) belongs to C1 or C2
for i=1:tM2
    if pC_v2(k, i) > pC_v2(k+1, i)
        t_est(i+tM1) = 0;
    else
        t_est(i+tM1) = 1;
    end;
end;

Ctest = confmat(t, t_est') % uses PRTools
err_test = 1-sum(diag(Ctest))/sum(Ctest(:)) % correct classification percentage

Ctrain = 0;

figure,
hold on
plot(log(pC_v1(1,:)), 'r');
plot(log(pC_v1(2,:)), 'b');
title('GMM3D test if V1 belongs to GMV1(red), GMV2(blue)');

figure,
hold on
plot(log(pC_v2(1,:)), 'r');
plot(log(pC_v2(2,:)), 'b');
title('GMM3D test if V2 belongs to GMV1(red), GMV2(blue)');

order = 30;
%hf = fir1(order, 0.1, rectwin(order+1));
hf = fir1(order, 0.01, hann(order+1));

fpC_v1_1 = filter(hf,1,pC_v1(1,:));
fpC_v1_2 = filter(hf,1,pC_v1(2,:));
figure,
hold on
plot(log(fpC_v1_1), '.r');
plot(log(fpC_v1_2), 'b');
title('GMM3D test LP filteret V1 belongs to GMV1(red), GMV2(blue)');
xlabel('MFCC sample');
ylabel('probability logarithmic');
    
fpC_v2_1 = filter(hf,1,pC_v2(1,:));
fpC_v2_2 = filter(hf,1,pC_v2(2,:));
figure,
hold on
plot(log(fpC_v2_1), 'r');
plot(log(fpC_v2_2), '.b');
title('GMM3D test LP filteret V2 belongs to GMV1(red), GMV2(blue)');
xlabel('MFCC sample');
ylabel('probability logarithmic');

