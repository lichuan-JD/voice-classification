
clear
addpath('C:\IHA\TINONS1\HMMall\HMM')
addpath('C:\IHA\TINONS1\HMMall\KPMtools')
addpath('C:\IHA\TINONS1\HMMall\KPMstats')

%Exercise 1 :
%In this exercise, Markov chains should be investigated. Experiment with “MarkovModels.m”. Try
%to change the transition matrix and the initial probability p(z0) - observe the changes in the output.
% As expected the probability to stay in a certain state follows the
% transition probability matrix A

%A = [0.9 0.1 0 0; 0 0.9 0.1 0; 0 0 0.9 0.1; 0 0 0 1] % left-right models - 4 states
A = [0.5 0.3 0.2 0; 0 0.5 0.5 0; 0 0 0.99 0.01; 0.05 0 0 0.95] % Markow model with forward and backward states
%A = [0.8 0.1 0.1 0; 0 0.5 0.5 0; 0 0 0.99 0.01; 0.05 0 0 0.95] % Markow model with forward and backward states
%A = rand(4); A = A ./ repmat(sum(A,2), 1, 4) % full trans-mat

N = 200; % number of samples

% generate markov chain
state(1) = 1;
for n = 1:N-1,
    p_trans = A(state(n), :); % transition-probability p(z_n+1 | z_n = current state)
    p_trans_cum = cumsum(p_trans); % cumulative probs
    p_rnd = rand; % generate random U(0,1) number
    state(n+1) = min(find(p_rnd < p_trans_cum)); % new state 
end
stem(state)

% try a few times..
for iter_id = 1:10,
    state(1) = 1;
    for n = 1:N-1,
        p_trans = A(state(n), :); % transition-probability p(z_n+1 | z_n = current state)
        p_trans_cum = cumsum(p_trans); % cumulative probs
        p_rnd = rand; % generate random U(0,1) number
        state(n+1) = min(find(p_rnd < p_trans_cum)); % new state
    end
    stem(state)
    pause
end





