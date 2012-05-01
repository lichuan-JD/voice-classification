function [voicebox_mfcc_dmfcc] = mfcc_func(WavIn, fs)

WavIn = sqrt(length(WavIn)) * WavIn / norm(WavIn); % WHITENING OF SOUND INPUT    
WavIn = WavIn + eps;    % To avoid numerical probs
nc = 12; % no. of cepstral coeffs (apart from 0'th coef)
n = 660; % length of frame
inc = 220; % increment = hop size (in number of samples)
p = floor(3*log(fs)) ;
voicebox_mfcc_dmfcc = melcepst(WavIn, fs, 'M0d',nc, p, n, inc);

%Bemærk : n og inc er i antal samples – dvs. skal evt. ændres i forhold til jeres samplerate fs.. 
%og voicebox_mfcc_dmfcc indeholder så både mfcc og delta_mfcc = mfcc(tid_n) – mfcc(tid_n-1)
