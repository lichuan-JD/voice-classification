load GMMtest_error
load ANNtest_error

figure, hold on,
plot(GMMtest_error, '*b');
plot(ANNtest_error, 'xr');
hold off;
title('ANN(*) and GMM(x) classifcation error vs. dimensions');
xlabel('dimensions');
ylabel('test error');