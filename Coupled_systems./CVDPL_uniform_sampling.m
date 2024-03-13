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
%% Uniform sampling
tic
NSR = 0.01;
if NSR ~= 0
    M = 100; % number of simulations
else
    M = 10;
end

if NSR == 0
    tol = 0.001;
elseif NSR == 0.001
    tol = 5;
else
    tol = 20;
end
envConstants.Lambda_train = 0.01;
% Weight for log(condnum)
envConstants.Lambda_condnum = 0.01;
% Weight for log(trace)
envConstants.Lambda_trace = 0.01;
envConstants.lambda = 0.1;
max_step = 500;
T0 = 0.004;
Ts = [T0 2*T0 2^2*T0 2^3*T0 2^4*T0 2^5*T0 2^6*T0];
dt = 0.004;
for k = 7:7
    Ts_cur = Ts(k);
    totalReward = zeros(M,1);
    best_sample_size = max_step;
    sample_size = zeros(M,1);
    train_time = zeros(M,1);
    condnum = zeros(M,1);
    Fnorm_error_last = zeros(M,1);
    trace_vec = zeros(M,1);
    reward = zeros(M,4);
    for i = 1:M
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
            % Add new data point
            idx = idx + Ts_cur/dt;
            XRL = [XRL; cx(idx,:)];
            dXRL = [dXRL; dx(idx,:)];
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
        train_time(i) = dt*idx;
        Theta = poolData(XRL,n,polyorder);
        condnum(i) = log(cond(Theta));
        trace_vec(i) = real(log(real(trace(inv(Theta'*Theta)))));
        Fnorm_error_last(i) = Fnorm_error_cur;
        if sample_size(i) < best_sample_size
            best_sample_size = sample_size(i);
        end
    end
 
    idx = find(sample_size < max_step);
    stat_mat = [sample_size(idx) Fnorm_error_last(idx) condnum(idx) trace_vec(idx) train_time(idx)];
    reward_stat = round([mean(reward(idx,:)); std(reward(idx,:))],4);
    elapsedTime = toc;
    stat_mean = [mean(stat_mat) length(idx)/M elapsedTime best_sample_size];
    stat_std = round(std(stat_mat),4);
    mean(totalReward(idx))
    std(totalReward(idx))
end
