
x = linspace(-5, 5, 1000);

% logistic sigmoid activation function = e^wx / (1 + e^wx)
figure
w = 1; w0 = 0;
y_log = (1./(1 + exp(-(w*x + w0)))); 
subplot(311), plot(x, y_log)

w = 4; w0 = 0;
y_log = (1./(1 + exp(-(w*x + w0)))); 
subplot(312), plot(x, y_log)

w = 10; w0 = 0;
y_log = (1./(1 + exp(-(w*x + w0)))); 
subplot(313), plot(x, y_log)

figure,
w = 2; w0 = 0;
y_log = (1./(1 + exp(-(w*x + w0)))); 
subplot(311), plot(x, y_log)

w = 4; w0 = 0;
y_log = (1./(1 + exp(-(w*x + w0)))); 
subplot(312), plot(x, y_log)

w = 4; w0 = 12;
y_log = (1./(1 + exp(-(w*x + w0)))); 
subplot(313), plot(x, y_log)


% generalisation -> softmax, y_k(x) = e^(w_k*x) / sum(e^w_k*x)




