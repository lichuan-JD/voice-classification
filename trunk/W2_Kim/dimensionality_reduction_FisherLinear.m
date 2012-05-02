
clear
close all

% MNIST handwritten digits data
%load('H:\Kurser_undervisning\TINONS1\DataSets\MNIST_HandwrittenDigits\mnist_all.mat')
load('.\mnist_all.mat')


X1 = double(train1);  % definition of our dataset
figure(1), imshow(reshape(X1(10,:), 28, 28)') % image 10

X2 = double(train2);  % definition of our dataset



%%%%%% Fisher Discriminant Analysis
N1 = size(X1,1);
mu1 = mean(X1);

N2 = size(X2,1);
mu2 = mean(X2);

% total scatter matrix
Sw = (X1 - repmat(mu1, N1, 1))'*(X1 - repmat(mu1, N1, 1)) + (X2 - repmat(mu2, N2, 1))'*(X2 - repmat(mu2, N2, 1));

% [v,d] = eig(Sw) ;
Swinv = inv(Sw + eye(size(Sw)));
w = Swinv*(mu2-mu1)';

% project to fisher "discriminant"
X1proj = X1*w;
X2proj = X2*w;

% plot histograms of the data in the new basis
ed1 = linspace(-5, 6,100)*10^-3;
[num1, bin1] = histc(X1proj, ed1);
subplot(211), bar(ed1, num1, 'histc')
[num2, bin2] = histc(X2proj, ed1);
subplot(212), bar(ed1, num2, 'histc')




%%%% Multiple Discriminant Analysis
Sw = zeros(size(Sw)); % nulstil Sw
Sb = zeros(size(Sw));
mu_tot = zeros(size(mu1));
Ntot = 0;
for i=1:5,
    X = double(eval(['train' num2str(i)]));
    mu = mean(X);
    N = size(X,1);
    Sw = Sw + (X - repmat(mu, N, 1))'*(X - repmat(mu, N, 1));
    mu_tot = mu_tot + N*mu;
    Ntot = Ntot + N;
end
mu_tot = mu_tot / Ntot;
for i=1:5,
    X = double(eval(['train' num2str(i)]));
    mu = mean(X);
    N = size(X,1);
    Sb = Sb + N*(mu - mu_tot)'*(mu - mu_tot);
end


Swinv = inv(Sw + eye(size(Sw))); % could change reg.parameter..
[v,d] = eig(Swinv*Sb);
%[v,d] = eig(Sb, Sw);
figure,
plot(log10(abs(diag(d))))

figure
% Two directions
W = v(:,1:2);
mycolors = {'r.', 'b.', 'g.', 'k.', 'c.'};
for i=1:5,
    X = double(eval(['train' num2str(i)]));
    Xnew = X(1:200,:)*W;
    scatter(Xnew(:,1), Xnew(:,2), mycolors{i}), hold on
end


figure
% Three directions
W = v(:,1:3);
mycolors = {'r.', 'b.', 'g.', 'k.', 'c.'};
for i=1:5,
    X = double(eval(['train' num2str(i)]));
    Xnew = X(1:200,:)*W;
    scatter3(Xnew(:,1), Xnew(:,2), Xnew(:,3), mycolors{i}), hold on
end

