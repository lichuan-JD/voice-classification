function [] = createplots(mfcc_in)

figure,
hold on
plot(mfcc_in(:,1))
plot(mfcc_in(:,2), 'r')
plot(mfcc_in(:,3), 'g')
plot(mfcc_in(:,4), 'y')
plot(mfcc_in(:,5), 'k')

