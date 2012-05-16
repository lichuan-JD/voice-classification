
clear 
addpath('C:\IHA\TINONS1\HMMall\HMM')
addpath('C:\IHA\TINONS1\HMMall\KPMtools')
addpath('C:\IHA\TINONS1\HMMall\KPMstats')

% sound data - 23 utterances of "HANS" and "GRETE"...
% (poor quality).. melcepst-feature vectors made from these..
load('hans_grete_data.mat');
soundsc(h{1}, fs)
soundsc(g{1}, fs)


N1 = 15; % number of training sequences..
Ntest = 8;
N_hiddenstates = 7; % number of hidden states
Ndim = 8; % dimensions (of feature vector)
Niter = 20; % max iteration of EM-algo.


% train hmm on class 1 ("hans")
% Train model on training data
% initial guess of parameters - randomly
prior0 = zeros(N_hiddenstates, 1); prior0(1) = 1; % start in state 1
transmat0 = rand(N_hiddenstates,N_hiddenstates);
% ensure left-right model - and has to pass through all states..
for i=1:N_hiddenstates,  
    for j=1:i-1,
        transmat0(i, j) = 0;
    end
    for j=i+2:N_hiddenstates,
        transmat0(i, j) = 0;
    end
end
transmat0 = mk_stochastic(transmat0);
sigma0 = repmat(eye(Ndim), [1 1 N_hiddenstates]); % covariance matrices
idx = randperm(size(hfeat_train{1}, 2));
for i=1:N_hiddenstates,
    mu0(:,i) = hfeat_train{1}(:, idx(i)); % choosing mu's randomly from data set..
end
% training - hmm with gaussian outputs
[LLh, prior1h, transmat1h, mu1h, sigma1h, mixmat1h] = ...
    mhmm_em(hfeat_train, prior0, transmat0, mu0, sigma0, [], 'max_iter', Niter);
 


% train hmm on class 2 ("grete")
% Train model on training data
% initial guess of parameters - randomly
prior0 = zeros(N_hiddenstates, 1); prior0(1) = 1; % start in state 1
transmat0 = rand(N_hiddenstates,N_hiddenstates);
% ensure left-right model - and has to pass through all states..
for i=1:N_hiddenstates,  
    for j=1:i-1,
        transmat0(i, j) = 0;
    end
    for j=i+2:N_hiddenstates,
        transmat0(i, j) = 0;
    end
end
transmat0 = mk_stochastic(transmat0);
sigma0 = repmat(eye(Ndim), [1 1 N_hiddenstates]); % covariance matrices
idx = randperm(size(gfeat_train{1}, 2));
for i=1:N_hiddenstates,
    mu0(:,i) = gfeat_train{1}(:, idx(i)); % choosing mu's randomly from data set..
end
% training - hmm with gaussian outputs
[LLg, prior1g, transmat1g, mu1g, sigma1g, mixmat1g] = ...
    mhmm_em(gfeat_train, prior0, transmat0, mu0, sigma0, [], 'max_iter', Niter);
 


% check values
transmat1g, transmat1h



%% VITERBI PATHS - e.g. for general word-recognition..
for seq_id = 1:10,
    outputs = gfeat_train{seq_id};
    for t=1:size(outputs,2),
        for i=1:N_hiddenstates,
            obslik(i, t) = gaussian_prob(outputs(:,t), mu1g(:,i), sigma1g(:,:,i));
        end
    end
    path = viterbi_path(prior1g, transmat1g, obslik);
    subplot(211), stem(path) % estimated hidden path
    subplot(212), plot(g{seq_id}), axis tight
    pause
end

for seq_id = 1:10,
    outputs = hfeat_train{seq_id};
    for t=1:size(outputs,2),
        for i=1:N_hiddenstates,
            obslik(i, t) = gaussian_prob(outputs(:,t), mu1h(:,i), sigma1h(:,:,i));
        end
    end
    path = viterbi_path(prior1h, transmat1h, obslik);
    subplot(211), stem(path) % estimated hidden path
    subplot(212), plot(h{seq_id}), axis tight
    pause
end





%% (Log) Likelihoods - used to classify between words..
% test HANS
for seq_id = 1:Ntest,
   
    outputs = hfeat_test{seq_id}; 
    for t=1:size(outputs,2),
        for i=1:N_hiddenstates,
            obslikg(i, t) = gaussian_prob(outputs(:,t), mu1g(:,i), sigma1g(:,:,i));
        end
    end
    [alphag, betag, gammag, llg(seq_id)] = fwdback(prior1g, transmat1g, obslikg, 'scaled', 1); % eg. compare ll with model with different pi, A, phi
    
    for t=1:size(outputs,2),
        for i=1:N_hiddenstates,
            obslikh(i, t) = gaussian_prob(outputs(:,t), mu1h(:,i), sigma1h(:,:,i));
        end
    end
    [alphah, betah, gammah, llh(seq_id)] = fwdback(prior1h, transmat1h, obslikh, 'scaled', 1);
    Class_label(seq_id) = llh(seq_id) > llg(seq_id); % 1 for Hans, 0 for Grete..
end


% test GRETE
for seq_id = 1:Ntest,    
    
    outputs = gfeat_test{seq_id}; 
    for t=1:size(outputs,2),
        for i=1:N_hiddenstates,
            obslikg(i, t) = gaussian_prob(outputs(:,t), mu1g(:,i), sigma1g(:,:,i));
        end
    end
    [alphag, betag, gammag, llg(seq_id+Ntest)] = fwdback(prior1g, transmat1g, obslikg, 'scaled', 1); % eg. compare ll with model with different pi, A, phi
    
    for t=1:size(outputs,2),
        for i=1:N_hiddenstates,
            obslikh(i, t) = gaussian_prob(outputs(:,t), mu1h(:,i), sigma1h(:,:,i));
        end
    end
    [alphah, betah, gammah, llh(seq_id+Ntest)] = fwdback(prior1h, transmat1h, obslikh, 'scaled', 1);
    Class_label(seq_id+Ntest) = llh(seq_id+Ntest) > llg(seq_id+Ntest); % 1 for Hans, 0 for Grete..
end

subplot(211), plot(llg, 'r'), hold on, plot(llh, 'b')
subplot(212)
stem(Class_label), axis([-1 2*Ntest+1 -1 2])


