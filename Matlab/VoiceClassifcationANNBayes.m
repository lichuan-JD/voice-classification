clear;
close all;

UsePCA_MDAFeatureReduction = 1 % 0=PCA, 1=MDA
UseClassificationMethod = 4 % 0=2D, 1=3D, 2 = ANN2D, 3 = ANN3D, 4 = Bayesian decision theory
UseSamples = 125000;

%% Setup and select training voice

% Use op as training set
[stereo1, fs] = wavread('OpKimC.wav'); % Recording voice 1
[stereo2, fs] = wavread('OpBjarkeC.wav'); % Recording voice 2
[Silence, fs] = wavread('Silence.wav');
 
% Silence part of voice
start2 = 1; 
end2 = start2+UseSamples-1;

% Analysing Op - training set
start1 = 1; % Op in voice 1
end1 = start1+UseSamples-1;
start3 = 1; % Op in voice 2
end3 = start3+UseSamples-1;

%% Setup of mel-cepstrum based on frequences

if fs == 11025
    n = 660; % length of frame for 11025 Hz
    inc = 220; % increment = hop size (in number of samples)
end;
if fs == 44100
    n = 2640; % length of frame for 44100 Hz
    inc = 1320; % increment = hop size (in number of samples) (default n/2) Windows XP
    %inc = 1000; % increment = hop size (in number of samples) (default n/2) Windows 7 
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
    if UseClassificationMethod == 1 % Linear 3D
        subSet = [1 3 2];
    else
    if UseClassificationMethod == 3 % ANN 3D - select features
        subSet = [1 2 3 4];
    else
    if UseClassificationMethod == 4 % Baysian decision theory
        subSet = [1 2];
    else
        subSet = [1 2 3];        
    end    
    end
    end
    [v1] = MultipleDiscriminantAnalysis(mfcc_dmfcc_y, mfcc_dmfcc_z, mfcc_dmfcc_w, subSet);
end

Ynew = mfcc_dmfcc_y*v1; % projecting onto the new basis
Znew = mfcc_dmfcc_z*v1; % projection on the same basis..
Wnew = mfcc_dmfcc_w*v1; % projection on the same basis..

%% Test set pattern classification on voice sound ned

% Use ned as test set
[stereo1_test, fs] = wavread('NedKimC.wav'); % Recording voice 1
[stereo2_test, fs] = wavread('NedBjarkeC.wav'); % Recording voice 2

% Analysing Ned - test set
start1_test = 1; % Ned in voice 1
end1_test = start1_test+UseSamples-1;
start3_test = 1; % Ned in voice 2
end3_test = start3_test+UseSamples-1;

yt = stereo1_test(start1_test:end1_test); % Select only one channel of time samples (Voice 1)
mfcc_dmfcc_yt = mfcc_func(yt,fs,n,inc); % Computes cepstrum MFCC features voice 1
wt = stereo2_test(start3_test:end3_test); % Select samples at other position in recording (Voice 2)
mfcc_dmfcc_wt = mfcc_func(w,fs,n,inc);

Ytnew = mfcc_dmfcc_yt*v1; % projecting on the same basis
Wtnew = mfcc_dmfcc_wt*v1; % projection on the same basis..

%% Plot projected features
if (size(subSet,1) > 2)
    figure(1);
    scatter3(Ynew(:,1), Ynew(:,2), Ynew(:,3), 'r.'); 
    hold on;
    scatter3(Znew(:,1), Znew(:,2), Znew(:,3), 'b.'); 
    scatter3(Wnew(:,1), Wnew(:,2), Wnew(:,3), 'g.'); 
    title('Projection of MFCC (Voice1 - red, Silence - blue, Voice2 - green)');
    xlabel('e1');
    ylabel('e2');
    zlabel('e3');
end

%% Classification of test set with 2 classes and 2 or 3 features

switch (UseClassificationMethod)
    case 0
        % 2D classification training set with 2 classes and 2 features
        [t_est, W] = linear2Dboundary(Ynew, Wnew); % training
        [tt_est, Wt] = linear2Dboundary(Ytnew, Wtnew); %test
    case 1
        % 3D classification training set with 2 classes and 3 features
        [t_est, W] = linear3Dboundary(Ynew, Wnew); % training
        [tt_est, Wt] = linear3Dboundary(Ytnew, Wtnew); %test
    case 2
        % 2D classification using Artificial Neural Networks
        [err_train, err_test] = ANN2D(Ynew, Ytnew, Wnew, Wtnew, Znew, 3); % 2 or 3 features
    case 3
        % 3D classification using Artificial Neural Networks
        [err_train, err_test] = ANN3D(Ynew, Ytnew, Wnew, Wtnew, Znew, size(subSet,2));
    case 4
        % Classification based on bayesian decision theory
        % Assuming a normal distribution of class features
        [t_est, Ctest] = gausianDiscriminant(Ynew, Ytnew, Wnew, Wtnew); % 2 features only
    otherwise
        % Invalid classification parameter specifier       
end
