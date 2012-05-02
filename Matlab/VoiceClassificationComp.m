clear;
close all;

%% Setup and select training voice 1 + 3
% [stereo1, fs] = wavread('OpKim.wav'); % Recording voice 1
% [stereo2, fs] = wavread('OpBjarke.wav'); % Recording voice 2
% 
% start1 = 27000; % P in voice 1
% end1 = 32000;
% start2 = 60000; % Silence part of voice 1
% end2 = 65000;
% start3 = 308000; % P in voice 2
% end3 = 313000;


%% Setup and select training voice 1 + 3 (Bjarke)

UsePCA_MDAFeatureReduction = 1 % 0=PCA, 1=MDA
UseLinear2D_3DBoundary = 1 % 0=2D, 1=3D
UseSamples = 125000;

% Use op as training set
[Op1, fs] = wavread('OpKimC.wav'); % Recording voice 1
[Op2, fs] = wavread('OpBjarkeC.wav'); % Recording voice 2
[Silence, fs] = wavread('Silence.wav');
 
start2 = 1; % Silence part of voice 1
end2 = start2+UseSamples-1;

% Analysing Op - training set
start1 = 1; % 2xOp in voice 1
end1 = start1+UseSamples-1;

start3 = 1; % 2xOp in voice 2
end3 = start3+UseSamples-1;

stereo1 = Op1;
stereo2 = Op2;


%% Setup of mel-cepstrum based on frequences
channel = 1; % Left or right channel (1,2)

if fs == 11025
    n = 660; % length of frame for 11025 Hz
    inc = 220; % increment = hop size (in number of samples)
end;
if fs == 44100
    n = 2640; % length of frame for 44100 Hz
    %inc = 1320; % increment = hop size (in number of samples) (default n/2)
    inc = 1000; % increment = hop size (in number of samples) (default n/2)
end;

%% PCA to find the 3 most dominating projected eigenvectors
y = stereo1(start1:end1); % Select only one channel of time samples (Voice 1)
mfcc_dmfcc_y = mfcc_func(y,fs,n,inc); % Computes cepstrum MFCC features voice 1
% Features of class 2 - another position in voice 1
z = Silence(start2:end2); % Select samples at other position in recording (Silence)
mfcc_dmfcc_z = mfcc_func(z,fs,n,inc);
% Features of class 3 - yet another position in voice 2
w = stereo2(start3:end3); % Select samples at other position in recording (Voice 2)
mfcc_dmfcc_w = mfcc_func(w,fs,n,inc);
features = size(mfcc_dmfcc_y,2)
samples = size(mfcc_dmfcc_y,1)

%% PCA or MDA eature reduction
if UsePCA_MDAFeatureReduction == 0
    % PCA feature reduction
    subSet = [1 3 2];
    [v1] = PrincipalComponentAnalysis(mfcc_dmfcc_y, subSet); % Sub set of principal components
else
    % MDA feature reduction
    subSet = [1 3 2];
    [v1] = MultipleDiscriminantAnalysis(mfcc_dmfcc_y, mfcc_dmfcc_z, mfcc_dmfcc_w, subSet);
end

Ynew = mfcc_dmfcc_y*v1; % projecting onto the new basis
Znew = mfcc_dmfcc_z*v1; % projection on the same basis..
Wnew = mfcc_dmfcc_w*v1; % projection on the same basis..

%% Restore voice 1, plot histogram
% new 3D representation / using feature vectors
Y = (mfcc_dmfcc_y - repmat(mean(mfcc_dmfcc_y), size(mfcc_dmfcc_y,1), 1))*v1; 
figure, hist(Y, 100) % Plotting histogram for three features
title('Histogram for features of Voice1 (Restored from e1, e2, e3)');

% Plot projected features
figure;
scatter3(Ynew(:,1), Ynew(:,2), Ynew(:,3), 'r.'); 
hold on;
scatter3(Znew(:,1), Znew(:,2), Znew(:,3), 'b.'); 
scatter3(Wnew(:,1), Wnew(:,2), Wnew(:,3), 'g.'); 
title('Projection of MFCC (Voice1 - red, Silence - blue, Voice2 - green)');
xlabel('e1');
ylabel('e2');
zlabel('e3');

%% Classification of training set with 2 classes and 2 or 3 features
if UseLinear2D_3DBoundary == 0
    % 2D classification training set with 2 classes and 2 features
    [t_est, W] = linear2Dboundary(Ynew, Wnew);
else
    % 3D classification training set with 2 classes and 3 features
    [t_est, W] = linear3Dboundary(Ynew, Wnew);
end

%% Test set pattern classification on voice sound ned

% Use ned as test set
[Ned1, fs] = wavread('NedKim.wav'); % Recording voice 1
[Ned2, fs] = wavread('NedBjarke.wav'); % Recording voice 2

% Analysing Ned - test set
start1_test = 75000; % Ned in voice 1
end1_test = start1_test+UseSamples-1;
start3_test = 75000; % Ned in voice 2
end3_test = start3_test+UseSamples-1;
stereo1_test = Ned1;
stereo2_test = Ned2;

yt = stereo1_test(start1_test:end1_test, channel); % Select only one channel of time samples (Voice 1)
mfcc_dmfcc_yt = mfcc_func(yt,fs,n,inc); % Computes cepstrum MFCC features voice 1
wt = stereo2_test(start3_test:end3_test, channel); % Select samples at other position in recording (Voice 2)
mfcc_dmfcc_wt = mfcc_func(w,fs,n,inc);

Ytnew = mfcc_dmfcc_yt*v1; % projecting onto the new basis
Wtnew = mfcc_dmfcc_wt*v1; % projection onto the new basis..

%% Classification of test set with 2 classes and 2 or 3 features
if UseLinear2D_3DBoundary == 0
    % 2D classification training set with 2 classes and 2 features
    [tt_est, Wt] = linear2Dboundary(Ytnew, Wtnew);
else
    % 3D classification training set with 2 classes and 3 features
    [tt_est, Wt] = linear3Dboundary(Ytnew, Wtnew);
end

%% Plotting MFCC values
 %figure, plot(y);
 %title('Voice1');
 %createplots(mfcc_dmfcc_y, fs, n, inc);
 %title('Voice1 MFCC');
 %plotmfcc(mfcc_dmfcc_y, fs, n, inc);
% 
 %figure, plot(z);
 %title('Silence');
 %createplots(mfcc_dmfcc_z)
 %title('Silence');
% 
 %figure, plot(w);
 %title('Voice2');
 %createplots(mfcc_dmfcc_w, fs, n, inc);
 %title('Voice2 MFCC');
 %plotmfcc(mfcc_dmfcc_w, fs, n, inc);
%