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
% start1 = 27000; % P in voice 1
% end1 = 32000;
% start2 = 60000; % Silence part of voice 1
% end2 = 65000;
% start3 = 308000; % P in voice 2
% end3 = 313000;


%% Setup and select training voice 1 + 3 (Bjarke)
[Op1, fs] = wavread('OpKim.wav'); % Recording voice 1
[Op2, fs] = wavread('OpBjarke.wav'); % Recording voice 2
 
start2 = 250000; % Silence part of voice 1
end2 = start2+25000;

[Ned1, fs] = wavread('NedKim.wav'); % Recording voice 1
[Ned2, fs] = wavread('NedBjarke.wav'); % Recording voice 2

op = 0;
if op == 1
 % Analysing Op
 start1 = 155000; % Op in voice 1
 end1 = start1+25000;
 start3 = 230000; % Op in voice 2
 end3 = start3+25000;
 stereo1 = Op1;
 stereo2 = Op2;
else   
 % Analysing Ned
 start1 = 75000; % Ned in voice 1
 end1 = start1+25000;
 start3 = 75000; % Ned in voice 2
 end3 = start3+25000;
 stereo1 = Ned1;
 stereo2 = Ned2;
end;

%% Setup of 
channel = 1; % Left or right channel (1,2)

if fs == 11025
    n = 660; % length of frame for 11025 Hz
    inc = 220; % increment = hop size (in number of samples)
end;
if fs == 44100
    n = 2640; % length of frame for 44100 Hz
    inc = 1320; % increment = hop size (in number of samples) (default n/2)
end;

%% PCA to find the 3 most dominating projected eigenvectors
y = stereo1(start1:end1, channel); % Select only one channel of time samples (Voice 1)
mfcc_dmfcc_y = mfcc_func(y,fs,n,inc); % Computes cepstrum MFCC features voice 1
% Features of class 2 - another position in voice 1
z = Op1(start2:end2, channel); % Select samples at other position in recording (Silence)
mfcc_dmfcc_z = mfcc_func(z,fs,n,inc);
% Features of class 3 - yet another position in voice 2
w = stereo2(start3:end3, channel); % Select samples at other position in recording (Voice 2)
mfcc_dmfcc_w = mfcc_func(w,fs,n,inc);
features = size(mfcc_dmfcc_y,2)


%% Plotting MFCC values
 figure, plot(y);
 title('Voice1');
 createplots(mfcc_dmfcc_y, fs, n, inc);
 title('Voice1 MFCC');
 plotmfcc(mfcc_dmfcc_y, fs, n, inc);
% 
 figure, plot(z);
 title('Silence');
 createplots(mfcc_dmfcc_z, fs, n, inc);
 title('Silence MFCC');
 plotmfcc(mfcc_dmfcc_z, fs, n, inc);
% 
 figure, plot(w);
 title('Voice2');
 createplots(mfcc_dmfcc_w, fs, n, inc);
 title('Voice2 MFCC');
 plotmfcc(mfcc_dmfcc_w, fs, n, inc);
%