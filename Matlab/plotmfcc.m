function [] = plotmfcc(mfcc_in, fs, n, inc)

 [nf,nc]=size(mfcc_in);
 t=((0:nf-1)*inc+(n-1)/2)/fs;
 %ci=(1:nc)-any(w=='0')-any(w=='E');
 ci=(1:nc);
 figure;
 imh = imagesc(t,ci,mfcc_in.');
 axis('xy');
 xlabel('Time (s)');
 ylabel('Mel-cepstrum coefficient');
 %map = (0:63)'/63;
 %colormap([map map map]);
 colormap(hsv(128));
 colorbar;

