%Exercise 2 :
%Experiment with Hidden Markov Models. Try to sample from a HMM using “generate_HMM.m”
%and estimate model parameters (training phase) using “HMMs.m”.

%generate_HMM.m
clear
close all
addpath('C:\IHA\TINONS1\HMMall\HMM')
addpath('C:\IHA\TINONS1\HMMall\KPMtools')
addpath('C:\IHA\TINONS1\HMMall\KPMstats')

N = 100; % number of samples in a sequence
Nsamples = 50; % number of training sequences..
N_hiddenstates = 3; % number hidden states
N_outputstates = 3; % number output states

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate data from HMM with gaussian emission probs.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sigma{1} = 1*eye(2); sigma{2} = 0.5*eye(2); sigma{3} = 0.5*eye(2); % spherical covariances
mu{1} = [-1 -1]; mu{2} = [5 7]; mu{3} = [0 8]; % mean values
A = [0.95 0.05 0; 0 0.95 0.05; 0 0 1] % transition matrix p(z_n | z_n-1)
for ex=1:Nsamples,    
    
    hidden_state(1) = 1; % start in state 1
    output_state(1) = 1;
    output_vec(1,:) = mvnrnd(mu{hidden_state(1)},sigma{hidden_state(1)},1); % draw from gaussian dist.
    
    for n = 2:N,
        p_trans = A(hidden_state(n-1), :); % transition-probability p(z_n | z_n-1 = current state)
        p_trans_cum = cumsum(p_trans); % cumulative probs
        p_rnd = rand; % generate random U(0,1) number
        hidden_state(n) = min(find(p_rnd < p_trans_cum)); % hidden state at time n
        
        output_vec(n,:) = mvnrnd(mu{hidden_state(n)},sigma{hidden_state(n)},1); % draw from gaussian dist.
        p = mvnpdf(output_vec(n,:),mu{hidden_state(n)},sigma{hidden_state(n)});
        if (p > 0.2)
            if (hidden_state(n) == 3)
                output_state(n) = 3;
            else    
                output_state(n) = 2;
            end
        else
            output_state(n) = 1;
        end
    end
    
    if ex < 5 || ex == 13
        figure,
        plot(output_vec(:,1), output_vec(:,2), 'x', 'MarkerSize', 10)
        hold on
        plot(output_vec(:,1), output_vec(:,2), 'r-')
        for k=1:3,
            muc=mu{k};
            text(muc(1),muc(2), num2str(k))
        end
    end

    data{ex} = output_state;
    hidden_data{ex} = hidden_state;
end

figure,
stem(data{13}) % example sequence..

% Train model on training data
% initial guess of parameters - randomly
prior1 = [1 0 0];
transmat1 = rand(N_hiddenstates,N_hiddenstates);
transmat1(2,1) = 0; transmat1(3,1) = 0; transmat1(3,2) = 0; % ensure left-right model  
transmat1 = mk_stochastic(transmat1)
obsmat1 = mk_stochastic(rand(N_hiddenstates,N_outputstates));

% train with EM algo.
[LL, prior2, transmat2, obsmat2] = dhmm_em(data, prior1, transmat1, obsmat1, 'max_iter', 30, 'adj_prior', 0)  % Discrete outputs  

%mhmm_em - alternative
% LEARN_MHMM Compute the ML parameters of an HMM with (mixtures of) Gaussians output using EM.

% Test model by estimating probability p(x=training sequence 13 | trained parameters pi, A, phi)
obslik = multinomial_prob(data{13}, obsmat2);
[alpha, beta, gamma, ll] = fwdback(prior2, transmat2, obslik, 'fwd_only', 1); % eg. compare ll with model with different pi, A, phi

% find most probable hidden path to have generated seq. 13..
obslik = multinomial_prob(data{13}, obsmat2);
path = viterbi_path(prior2, transmat2, obslik);
figure,
subplot(211), stem(path) % estimated hidden path
subplot(212), stem(hidden_data{13}), axis tight % true path (since generated data..)

