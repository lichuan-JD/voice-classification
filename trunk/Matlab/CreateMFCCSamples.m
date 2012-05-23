function [mfcc_voice1, mfcc_voice2, mfcc_silence] = CreateMFCCSamples(PlotMFCC, Pause, Start, End)

% Reads all sound recordings and create MFCC sample feature vectors
mfcc_voice1 = [];
mfcc_voice2 = [];
mfcc_silence = [];

for SamplesToLoad = Start:End 
    
    % Use op as training set
    switch (SamplesToLoad)
        case 0
            [stereo1, fs] = wavread('OpBjarkeC.wav'); % Recording voice 2
            [stereo2, fs] = wavread('OpKimC.wav'); % Recording voice 1
            [Silence, fs] = wavread('Silence.wav');
            UseSamples = 125000;
        case 1
            [stereo1, fs] = wavread('NedBjarkeC.wav'); % Recording voice 2
            [stereo2, fs] = wavread('NedKimC.wav'); % Recording voice 1
            [Silence, fs] = wavread('Silence.wav');
            UseSamples = 125000;
        case 2
            [stereo1, fs] = wavread('Speech1_1.wav'); % Recording voice 1
            [stereo2, fs] = wavread('Speech2_1.wav'); % Recording voice 2
            [Silence, fs] = wavread('Silence2.wav');
            UseSamples = 250000;
        case 3
            [stereo1, fs] = wavread('Speech1_2.wav'); % Recording voice 1 (Same text as 1)
            [stereo2, fs] = wavread('Speech2_2.wav'); % Recording voice 2
            [Silence, fs] = wavread('Silence2.wav');
            UseSamples = 250000;
        case 4
            [stereo1, fs] = wavread('Speech1_A.wav'); % Recording voice 1 (Very different text)
            [stereo2, fs] = wavread('Speech2_A.wav'); % Recording voice 2
            [Silence, fs] = wavread('Silence2.wav');
            UseSamples = 250000;
        case 5
            [stereo1, fs] = wavread('Speech1_B.wav'); % Recording voice 1 (Similar text as 1)
            [stereo2, fs] = wavread('Speech2_B.wav'); % Recording voice 2
            [Silence, fs] = wavread('Silence2.wav');
            UseSamples = 250000;
    end
    
    % Silence part of voice
    start2 = 1;
    end2 = start2+UseSamples-1;
    
    % Speech part of voice
    start1 = 1; % Speach voice 1
    end1 = start1+UseSamples-1;
    start3 = 1; % Speach voice 2
    end3 = start3+UseSamples-1;
    
    % Setup of mel-cepstrum based on sample frequence
    if fs == 11025
        n = 660; % length of frame for 11025 Hz
        inc = 220; % increment = hop size (in number of samples)
    end;
    if fs == 44100
        %n = 2640; % length of frame for 44100 Hz 60 ms of voice
        %inc = 1320; % increment = hop size (in number of samples) (default n/2) Windows XP
        n = 1320; % length of frame for 44100 Hz 30 ms of voice 
        inc = 660; % increment = hop size (in number of samples) (default n/2) Windows XP
        %inc = 1000; % increment = hop size (in number of samples) (default n/2) Windows 7
    end;
    
    % Features of class 1 - voice 1
    y = stereo1(start1:end1); % Select only one channel of time samples (Voice 1)
    mfcc_y = mfcc_func(y,fs,n,inc); % Computes cepstrum MFCC features voice 1
    
    % Features of class 2 - silence
    z = Silence(start2:end2); % Select samples at other position in recording (Silence)
    mfcc_z = mfcc_func(z,fs,n,inc);
    
    % Features of class 3 - voice 2
    w = stereo2(start3:end3); % Select samples at other position in recording (Voice 2)
    mfcc_w = mfcc_func(w,fs,n,inc);
    
    mfcc_voice1 = [mfcc_voice1; mfcc_y]; % Add to feature vector
    mfcc_voice2 = [mfcc_voice2; mfcc_w];
    mfcc_silence = [mfcc_silence; mfcc_z];
    
    if PlotMFCC > 0 
        %% Plotting MFCC values
        figure, plot(y);
        title('Voice1');
        createplots(mfcc_y, fs, n, inc);      
        title('Voice1 MFCC');
        if PlotMFCC > 1
            plotmfcc(mfcc_y, fs, n, inc);
        end
        %
        figure, plot(z);
        title('Silence');
        createplots(mfcc_z, fs, n, inc);
        title('Silence MFCC');
        if PlotMFCC > 1
            plotmfcc(mfcc_z, fs, n, inc);
        end
        %
        figure, plot(w);
        title('Voice2');
        createplots(mfcc_w, fs, n, inc);
        title('Voice2 MFCC');
        if PlotMFCC > 1
            plotmfcc(mfcc_w, fs, n, inc);
        end
        %
        if Pause == 1
            pause;
        end
    end

end
