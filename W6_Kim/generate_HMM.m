 
clear

N = 100; % number of samples in a sequence

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate data from HMM with gaussian emission probs.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sigma{1} = 1*eye(2); sigma{2} = 0.5*eye(2); sigma{3} = 0.5*eye(2); % spherical covariances
mu{1} = [-1 -1]; mu{2} = [5 7]; mu{3} = [0 8]; % mean values
A = [0.95 0.05 0; 0 0.95 0.05; 0 0 1] % transition matrix p(z_n | z_n-1)
hidden_state(1) = 1; % start in state 1
output_vec(1,:) = mvnrnd(mu{hidden_state(1)},sigma{hidden_state(1)},1); % draw from gaussian dist.
for n = 2:N,
    p_trans = A(hidden_state(n-1), :); % transition-probability p(z_n | z_n-1 = current state)
    p_trans_cum = cumsum(p_trans); % cumulative probs
    p_rnd = rand; % generate random U(0,1) number
    hidden_state(n) = min(find(p_rnd < p_trans_cum)); % hidden state at time n
    
    output_vec(n,:) = mvnrnd(mu{hidden_state(n)},sigma{hidden_state(n)},1); % draw from gaussian dist.
end

plot(output_vec(:,1), output_vec(:,2), 'x', 'MarkerSize', 10)
hold on
plot(output_vec(:,1), output_vec(:,2), 'r-')
for k=1:3, 
    muc=mu{k};
    text(muc(1),muc(2), num2str(k))
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate data from diskrete HMM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% p_init = [1 0 0];
A = [0.95 0.05 0; 0 0.95 0.05; 0 0 1] % left-right models - 3 hidden states
phi = [0.5 0.5; 0.9 0.1; 0.1 0.9] % hidden state 1 has 50% chance to give output state 1 or 2, hidden state 2 has 90% chance to give output state 1 etc..
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
subplot(211), stem(hidden_state)
subplot(212), stem(output_state)






