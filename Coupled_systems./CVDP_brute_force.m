clear all, close all, clc
%% Generate Data
n = 4;
% Initialize parameters
mu1 = 5;
mu2 = 4;
c1 = 0.005;
c2 = 1;
F = 5;
tau_fast = 0.2;
tau_slow = F*tau_fast;
x0 = [2,0,0,2];
par_vec = [mu1,mu2,tau_fast,tau_slow,c1,c2];

% Integrate
dt = 0.04;
tspan = [dt:dt:500*dt];
N = length(tspan);
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[t,x] = ode45(@(t,x) coupled_vdp(t,x,par_vec),tspan,x0,options);
% Compute derivatives
dx_true = zeros(length(x),n);
for i = 1:length(x)
    dx_true(i,:) = coupled_vdp(0,x(i,:),par_vec);
end
% Pool Data  (i.e., build library of nonlinear time series)
polyorder = 3;
Theta = poolData(x,n,polyorder);
m = size(Theta,2);

% Compute Sparse regression: sequential least squares
lambda = 0.1;      % lambda is our sparsification knob.
Xi_true = sparsifyDynamics(Theta,dx_true,lambda,n);
poolDataLIST({'x1','x2','x3','x4'},Xi_true,n,polyorder);
%% Brute force search

NSR = 0.01;
MM = 1;
% if NSR ~= 0
%     MM = 10; % number of simulations
% else
%     MM = 1;
% end

dt = 0.004;
if NSR == 0
    tol = 0.001;
elseif NSR == 0.001
    tol = 4;
else
    tol = 15;
end
% search_range = 20*dt:10*dt:100*dt;
envConstants.Lambda_train = 0.01;
% Weight for log(condnum)
envConstants.Lambda_condnum = 0.01;
% Weight for log(trace)
envConstants.Lambda_trace = 0.01;
envConstants.lambda = 0.1;

% The lower limit for Ts
Ts_low = 0.004;
% The upper limit for Ts
Ts_high = 0.32;
M = 10000;
max_step = 500;
T0 = 0.004;
p = [1/3 1/3 1/3];
best_sequence = zeros(MM,max_step);
tic
stat_mat = zeros(MM,5);
best_reward = zeros(MM,1);
[InitialObservation, LoggedSignals] = ResetFunction_CVDP(NSR);
cx = LoggedSignals.cx;
dx = LoggedSignals.dx;
for ii = 1:4
    tic
    best_sample_size = max_step;
    sample_size = zeros(M,1);
    train_time = zeros(M,1);
    condnum = zeros(M,1);
    Fnorm_error_last = zeros(M,1);
    trace_vec = zeros(M,1);
    reward = zeros(M,1);
    action_seq = zeros(MM,max_step);
    Ts_init = 0.016;
    for i = 1:M
        XRL = LoggedSignals.Xcur;
        dXRL = LoggedSignals.dXcur;
        sampled_data_idx = [];
        tmp_seq = []; % action sequence   
        Ts = Ts_init;
        k = 1;
        tnow = dt;
        for j = 1:k
            tnow = tnow + Ts_init;
        end
        idx = (tnow/dt);
        Fnorm_error_cur = 100;
        while Fnorm_error_cur > tol && length(XRL) <= 500
            % Randomize the action
            r = mnrnd(1,p);
            action = find(r == 1);
            tmp = (action == 3 && Ts < Ts_high)*Ts*2 + (action == 2)*Ts + ...
                (action == 1 && Ts > Ts_low)*Ts/2;
            Ts = (tmp == 0)*Ts + (tmp ~= 0)*tmp;
            if action == 3 && tmp ~= 0 
                tmp_seq  = [tmp_seq  3];
            elseif action == 2 || tmp == 0
                tmp_seq  = [tmp_seq  2];
            else
               tmp_seq  = [tmp_seq  1];
            end
    
            % Add new data point
            tnow = tnow + Ts;
            XRL = [XRL; cx(ceil(tnow/dt),:)];
            dXRL = [dXRL; dx(ceil(tnow/dt),:)];
            sampled_data_idx(end+1) = ceil(tnow/dt);
            Theta = poolData(XRL,n,polyorder);
            Xi_hat = sparsifyDynamics(Theta,dXRL,envConstants.lambda,n);
            Fnorm_error_cur = norm(abs(Xi_true)-abs(Xi_hat),'fro')^2;
            % Update reward
            tmp1 = log(cond(Theta));
            tmp2 = real(log(real(trace(inv(Theta'*Theta)))));
            switch ii
                case 1
                    reward(i) = reward(i)-envConstants.Lambda_condnum*tmp1;
                case 2
                    reward(i) =  reward(i)-envConstants.Lambda_trace*tmp2;
                case 3
                    reward(i) =  reward(i)-envConstants.Lambda_train*tnow;
                case 4
                    reward(i) = reward(i)-envConstants.Lambda_trace*tmp2-envConstants.Lambda_condnum*tmp1-envConstants.Lambda_train*dt*idx;
            end 
        end
        tmp_seq(length(tmp_seq)+1:max_step) = 0;
        % Evaluate the policy
        action_seq(i,:) = tmp_seq;
        sample_size(i) = length(XRL);
        train_time(i) = tnow;
        Theta = poolData(XRL,n,polyorder);
        condnum(i) = log(cond(Theta));
        trace_vec(i) = real(log(real(trace(inv(Theta'*Theta)))));
        Fnorm_error_last(i) = Fnorm_error_cur;
    end
    % Select the best policy
    selected_idx = find(~isnan(Fnorm_error_last));
    tmp = [sample_size(selected_idx) Fnorm_error_last(selected_idx) condnum(selected_idx) trace_vec(selected_idx) train_time(selected_idx)];
    % switch ii
    %     case 1
    %         [max_t,idx] = min(condnum(selected_idx));
    %     case 2
    %         [max_t,idx] = max(trace_vec(selected_idx));
    %     case 3
    %         [max_t,idx] = min(train_time(selected_idx));
    %     case 4
    %         [max_t,idx] = max(reward(selected_idx));
    % end
    [max_t,idx] = max(reward(selected_idx));
    reward1 = reward(selected_idx);
    action_seq1 = action_seq(selected_idx,:);
    best_reward(ii) = reward1(idx);
    best_sequence(ii,:) = action_seq1(idx,:);
    tmp_vec = tmp(idx,:);
    stat_mat(ii,:) = tmp_vec;
    elapsedTime(ii) = toc;
end
% idx = find(sample_size < max_step & ~isnan(Fnorm_error_last));
% stat_mat = [sample_size(idx) Fnorm_error_last(idx) condnum(idx) trace_vec(idx) train_time(idx)];
% reward_stat = round([mean(reward(idx,:)); std(reward(idx,:))],4);
% elapsedTime = toc;
% stat_mean = [mean(stat_mat) length(idx)/M elapsedTime best_sample_size];
% stat_std = round(std(stat_mat),4);
% 
% mean(totalReward(idx))
% std(totalReward(idx))
