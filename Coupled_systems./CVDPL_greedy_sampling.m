clear all, close all, clc
%% Generate Data
n = 5;
% Initialize parameters
mu = 5;
c1 = 0.01;
c2 = 10;
F = 5;
sigma = 10; rho = 28; beta = 8/3;
tau_fast = 0.2;
tau_slow = F*tau_fast;
x0 = [2 0 -8 8 27]';

% Integrate
dt = 0.04;
tspan = [dt:dt:200];
par_vec = [mu,sigma,rho,beta,tau_fast,tau_slow,c1,c2];
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[t,x]=ode45(@(t,x) coupled_vdp_lorenz(t,x,par_vec),tspan,x0,options);
% Compute derivative using Total Variation Regularized Numerical Differentiation (TVDiff)
for i=1:length(x)
    dx_true(i,:) = coupled_vdp_lorenz(0,x(i,:),par_vec);
end
% Pool Data  (i.e., build library of nonlinear time series)
polyorder = 3;
Theta = poolData(x,n,polyorder);
m = size(Theta,2);

% Compute Sparse regression: sequential least squares
lambda = 0.1;      % lambda is our sparsification knob.
Xi_true = sparsifyDynamics(Theta,dx_true,lambda,n);
%% Greedy sampling
opt = 1; % select optimality criterion 1, 2, or 3

NSR = 0.001;
if NSR ~= 0
    M = 20; % number of simulations
else
    M = 10;
end

dt = 0.004;
if NSR == 0
    tol = 0.001;
    search_range = 50*dt;
elseif NSR == 0.001
    tol = 5;
    search_range = 50*dt;
else
    tol = 20;
    search_range = 60*dt;
end
search_range = 20*dt:10*dt:80*dt;
envConstants.Lambda_train = 0.01;
% Weight for log(condnum)
envConstants.Lambda_condnum = 0.01;
% Weight for log(trace)
envConstants.Lambda_trace = 0.01;
envConstants.lambda = 0.1;
max_step = 500;
T0 = 0.004;
final_mat = [];
best_sequence = [];
for ii = 1:length(search_range)
    tic
    totalReward = zeros(M,1);
    best_sample_size = max_step;
    sample_size = zeros(M,1);
    train_time = zeros(M,1);
    condnum = zeros(M,1);
    Fnorm_error_last = zeros(M,1);
    trace_vec = zeros(M,1);
    reward = zeros(M,4);
    for i = 1:M
        sampled_data_idx = [];
        [InitialObservation, LoggedSignals] = ResetFunction_CVDPL(NSR);
        cx = LoggedSignals.cx;
        dx = LoggedSignals.dx;
        XRL = LoggedSignals.Xcur;
        dXRL = LoggedSignals.dXcur;
        Ts_init = 0.016;
        k = 1;
        tnow = dt;
        for j = 1:k
            tnow = tnow + Ts_init;
        end
        idx = (tnow/dt);
        Fnorm_error_cur = 100;
        while Fnorm_error_cur > tol && length(XRL) <= 500
            % find the data point to sample
            n_points = length(dt:dt:search_range(ii));
            I_mat_trace = zeros(n_points,1); % information matrix trace
            I_mat_det = zeros(n_points,1); % information matrix determinant
            I_mat_eig = zeros(n_points,1); % information matrix smallest eigenvalue
            
            count = 0;
            for t = dt:dt:search_range(ii)
                count = count + 1;
                % sample at t_now + t
                X_tmp = [XRL; cx(ceil((tnow+t)/dt),:)];
                Theta = poolData(X_tmp,n,polyorder);
                I_mat = inv(Theta'*Theta);% information matrix
                I_mat_trace(count) = trace(I_mat);
                % I_mat_det(count) = det(I_mat);
                % I_mat_eig(count) = min(eig(I_mat));
            end   
    
            % Greedy sampling and plot the distribution
            switch opt
            case 1 % A-optimality (trace)
                [~,m_idx] = max(I_mat_trace);
            case 2 % D-optimality 
                [~,m_idx] = max(I_mat_det);
            case 3 % E-optimality
                [~,m_idx] = max(I_mat_eig);
            end
            % Add new data point
            tnow = tnow + m_idx*dt;
            XRL = [XRL; cx(ceil(tnow/dt),:)];
            dXRL = [dXRL; dx(ceil(tnow/dt),:)];
            sampled_data_idx(end+1) = ceil(tnow/dt);
            Theta = poolData(XRL,n,polyorder);
            Xi_hat = sparsifyDynamics(Theta,dXRL,envConstants.lambda,n);
            Fnorm_error_cur = norm(abs(Xi_true)-abs(Xi_hat),'fro')^2;
            % Update reward
            tmp1 = log(cond(Theta));
            tmp2 = real(log(real(trace(inv(Theta'*Theta)))));
            reward(i,1) = reward(i,1)-envConstants.Lambda_condnum*tmp1;
            reward(i,2) =  reward(i,2)-envConstants.Lambda_trace*tmp2;
            reward(i,3) =  reward(i,3)-envConstants.Lambda_train*dt*idx;
            reward(i,4) = reward(i,4)-envConstants.Lambda_trace*tmp2-envConstants.Lambda_condnum*tmp1-envConstants.Lambda_train*dt*idx;
        end
    
        % Evaluate the policy
        sample_size(i) = length(XRL);
        train_time(i) = tnow;
        Theta = poolData(XRL,n,polyorder);
        condnum(i) = log(cond(Theta));
        trace_vec(i) = real(log(real(trace(inv(Theta'*Theta)))));
        Fnorm_error_last(i) = Fnorm_error_cur;
        if sample_size(i) < best_sample_size && ~isnan(Fnorm_error_cur)
            best_sample_size = sample_size(i);
            best_sequence = sampled_data_idx;
        end
    end
    idx = find(sample_size < max_step & ~isnan(Fnorm_error_last));
    stat_mat = [sample_size(idx) Fnorm_error_last(idx) condnum(idx) trace_vec(idx) train_time(idx)];
    reward_stat = round([mean(reward(idx,:)); std(reward(idx,:))],4);
    elapsedTime = toc;
    stat_mean = [mean(stat_mat) length(idx)/M elapsedTime best_sample_size];
    stat_std = round(std(stat_mat),4);
    final_mat(ii,:) = stat_mean;
end


mean(totalReward(idx))
std(totalReward(idx))
