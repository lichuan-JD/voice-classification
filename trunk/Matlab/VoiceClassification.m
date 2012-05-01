clear;
close all;

%% Setup and select training voice 1
% [stereo1, fs] = wavread('OpNedKim.wav');
% stereo2 = stereo1;
% start1 = 80000;
% end1 = 82000;
% start2 = 85000;
% end2 = 87000;
% start3 = 88000;
% end3 = 90000;

%% Setup and select training voice 1 + 3
% [stereo1, fs] = wavread('OpKim.wav'); % Recording voice 1
% [stereo2, fs] = wavread('OpBjarke.wav'); % Recording voice 2
% 
% start1 = 158000; % P in voice 1
% end1 = start1+25000;
% start2 = 250000; % Silence part of voice 1
% end2 = start2+25000;
% start3 = 220000; % P in voice 2
% end3 = start3+25000;

[stereo1, fs] = wavread('NedKim.wav'); % Recording voice 1
[stereo2, fs] = wavread('NedBjarke.wav'); % Recording voice 2

start1 = 82000; % P in voice 1
end1 = start1+2000;
start2 = 50000; % Silence part of voice 1
end2 = start2+2000;
start3 = 214000; % P in voice 2
end3 = start3+2000;


channel = 1; % Left or right channel (1,2)

%% PCA to find the 3 most dominating projected eigenvectors
y = stereo1(start1:end1, channel); % Select only one channel of time samples (Voice 1)
mfcc_dmfcc_y = mfcc_func(y,fs); % Computes cepstrum MFCC features voice 1
% Features of class 2 - another position in voice 1
z = stereo1(start2:end2, channel); % Select samples at other position in recording (Silence)
mfcc_dmfcc_z = mfcc_func(z,fs);
% Features of class 3 - yet another position in voice 2
w = stereo2(start3:end3, channel); % Select samples at other position in recording (Voice 2)
mfcc_dmfcc_w = mfcc_func(w,fs);
features = size(mfcc_dmfcc_y,2)

%subSet = [1 3 2];
subSet = [1 2 3];

%% PCA feature reduction
[v1] = PrincipalComponentAnalysis(mfcc_dmfcc_y, subSet); % Sub set of principal components

%% MDA feature reduction
%[v1] = MultipleDiscriminantAnalysis(mfcc_dmfcc_y, mfcc_dmfcc_z, mfcc_dmfcc_w, subSet);

Ynew = mfcc_dmfcc_y*v1; % projecting onto the new basis
Znew = mfcc_dmfcc_z*v1; % projection on the same basis..
Wnew = mfcc_dmfcc_w*v1; % projection on the same basis..

%% Restore voice 1, plot histogram
% new 3D representation / using feature vectors
Y = (mfcc_dmfcc_y - repmat(mean(mfcc_dmfcc_y), size(mfcc_dmfcc_y,1), 1))*v1; 
figure, hist(Y, 100) % Plotting histogram for three features
title('Histogram for features of Voice1 (Restored from e1, e2, e3)');

%% Plot projected features
figure;
scatter3(Ynew(:,1), Ynew(:,2), Ynew(:,3), 'r.'); 
hold on;
scatter3(Znew(:,1), Znew(:,2), Znew(:,3), 'b.'); 
scatter3(Wnew(:,1), Wnew(:,2), Wnew(:,3), 'g.'); 
title('Projection of MFCC (Voice1 - red, Silence - blue, Voice2 - green)');
xlabel('e1');
ylabel('e2');
zlabel('e3');

%% 2D example - classification training set with 2 classes and 2 features
%[t_est, W] = linear2Dboundary(Ynew, Wnew);

%% 3D example - classification training set with 2 classes and 3 features
[t_est, W] = linear3Dboundary(Ynew, Wnew);

%% Plotting MFCC values
figure, plot(y);
title('Voice1');
createplots(mfcc_dmfcc_y)
title('Voice1 MFCC (1-5)');

figure, plot(z);
title('Silence');
createplots(mfcc_dmfcc_z)
title('Silence');

figure, plot(w);
title('Voice2');
createplots(mfcc_dmfcc_w)
title('Voice2 MFCC (1-5)');
