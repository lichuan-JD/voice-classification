function [voicebox_mfcc_dmfcc] = mfcc_func(WavIn, fs, n, inc)

WavIn = sqrt(length(WavIn)) * WavIn / norm(WavIn); % WHITENING OF SOUND INPUT    
WavIn = WavIn + eps;    % To avoid numerical probs
nc = 12; % no. of cepstral coeffs (apart from 0'th coef)
p = floor(3*log(fs)) ;
voicebox_mfcc_dmfcc = melcepst(WavIn, fs, 'N',nc, p, n, inc); %'M0d'
%Bemærk : n og inc er i antal samples – dvs. skal evt. ændres i forhold til jeres samplerate fs.. 
%og voicebox_mfcc_dmfcc indeholder så både mfcc og delta_mfcc = mfcc(tid_n) – mfcc(tid_n-1)
