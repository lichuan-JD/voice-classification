function [W] = MultipleDiscriminantAnalysis(class1, class2, class3, subSet)

mu = mean(class1);
N = size(class1,1);
Sw = (class1 - repmat(mu, N, 1))'*(class1 - repmat(mu, N, 1));
Sw = zeros(size(Sw)); % nulstil Sw
Sb = zeros(size(Sw));
mu_tot = zeros(size(mu));
Ntot = 0;
for i=1:3,
    X = double(eval(['class' num2str(i)]));
    mu = mean(X);
    N = size(X,1);
    Sw = Sw + (X - repmat(mu, N, 1))'*(X - repmat(mu, N, 1));
    mu_tot = mu_tot + N*mu;
    Ntot = Ntot + N;
end
mu_tot = mu_tot / Ntot;
for i=1:3,
    X = double(eval(['class' num2str(i)]));
    mu = mean(X);
    N = size(X,1);
    Sb = Sb + N*(mu - mu_tot)'*(mu - mu_tot);
end


Swinv = inv(Sw + eye(size(Sw))); % could change reg.parameter..
[v,d] = eig(Swinv*Sb);
%[v,d] = eig(Sb, Sw);
W = v(:, subSet);
