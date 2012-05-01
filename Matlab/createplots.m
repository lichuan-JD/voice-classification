function [] = createplots(mfcc_in, fs, n, inc)

figure,
hold on
plot(mfcc_in(1,:), 'b')
plot(mfcc_in(2,:), 'r')
plot(mfcc_in(3,:), 'g')
plot(mfcc_in(4,:), 'y')
plot(mfcc_in(5,:), 'k')
plot(mfcc_in(6,:), 'b')
plot(mfcc_in(7,:), 'r')
plot(mfcc_in(8,:), 'g')
plot(mfcc_in(9,:), 'y')
plot(mfcc_in(10,:),'k')
plot(mfcc_in(11,:), 'b')
plot(mfcc_in(12,:), 'r')
plot(mfcc_in(13,:), 'g')
plot(mfcc_in(14,:), 'y')
plot(mfcc_in(15,:), 'k')
plot(mfcc_in(16,:), 'b')
plot(mfcc_in(17,:), 'r')
xlabel('Mel-cepstrum coefficients (1-12)');
ylabel('Value for 17 delta times');

