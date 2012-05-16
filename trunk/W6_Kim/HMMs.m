
clear 

Nsamples = 50; % number of training sequences..

N_hiddenstates = 3; % number hidden states
N_outputstates = 2; % number output states
A = [0.95 0.05 0; 0 0.95 0.05; 0 0 1]; % left-right models - 3 hidden states
phi = [0.5 0.5; 0.9 0.1; 0.1 0.9]; % hidden state 1 has 50% chance to give output state 1 or 2, hidden state 2 has 90% chance to give output state 1 etc.
N = 100; % number of samples
% Generate training sequences..
for ex=1:Nsamples,    
    hidden_state(1) = 1; % start in state 1
    for n = 1:N,
        p_trans = A(hidden_state(n), :); % transition-probability p(z_n+1 | z_n = current state)
        p_trans_cum = cumsum(p_trans); % cumulative probs
        p_rnd = rand; % generate random U(0,1) number
        hidden_state(n+1) = min(find(p_rnd < p_trans_cum)); % hidden state at time n+1
        
        p_emis = phi(hidden_state(n), :); % emission-prob (p(x | z_n = current state)) - discrete output
        p_emis_cum = cumsum(p_emis);
        p_rnd = rand;
        output_state(n) = min(find(p_rnd < p_emis_cum)); % output state at time n
    end
    data{ex} = output_state;
    hidden_data{ex} = hidden_state;
end
stem(data{13}) % example sequence..


% Train model on training data
% initial guess of parameters - randomly
prior1 = [1 0 0];
transmat1 = rand(N_hiddenstates,N_hiddenstates);
transmat1(2,1) = 0; transmat1(3,1) = 0; transmat1(3,2) = 0; % ensure left-right model  
transmat1 = mk_stochastic(transmat1)
obsmat1 = mk_stochastic(rand(N_hiddenstates,N_outputstates));

% train with EM algo.
[LL, prior2, transmat2, obsmat2] = dhmm_em(data, prior1, transmat1, obsmat1, 'max_iter', 30, 'adj_prior', 0)    


% estimate probability p(x=training sequence 13 | trained parameters pi, A,
% phi)
obslik = multinomial_prob(data{13}, obsmat2);
[alpha, beta, gamma, ll] = fwdback(prior2, transmat2, obslik, 'fwd_only', 1); % eg. compare ll with model with different pi, A, phi

% find most probable hidden path to have generated seq. 13..
obslik = multinomial_prob(data{13}, obsmat2);
path = viterbi_path(prior2, transmat2, obslik);
subplot(211), stem(path) % estimated hidden path
subplot(212), stem(hidden_data{13}), axis tight % true path (since generated data..)
