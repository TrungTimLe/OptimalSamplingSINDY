clear all, close all, clc
%% Generate Data
Beta = [10; 28; 8/3]; % Lorenz's parameters (chaotic)
n = 3; %number of state
x0 = [-8; 8; 27];  % Initial condition

% Integrate
dt = 0.04;
tspan = [dt:dt:40];
N = length(tspan);
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[t,x] = ode45(@(t,x) lorenz(t,x,Beta),tspan,x0,options);
% Compute derivative using Total Variation Regularized Numerical Differentiation (TVDiff)
for i=1:length(x)
    dx_true(i,:) = lorenz(0,x(i,:),Beta);
end
% Pool Data  (i.e., build library of nonlinear time series)
polyorder = 3;
Theta = poolData(x,n,polyorder);
m = size(Theta,2);

% Compute Sparse regression: sequential least squares
lambda = 0.1;      % lambda is our sparsification knob.
Xi_true = sparsifyDynamics(Theta,dx_true,lambda,n);
poolDataLIST({'x','y','z'},Xi_true,n,polyorder);
%% Generate data for best policy
rng(2)
n = 3; %number of state
x0 =[-8; 8; 27];  % Initial condition
dt = 0.004; % delta t
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
Tmax = 1000;
tspan = [dt:dt:Tmax];
N = length(tspan);
[t,x] = ode45(@(t,x) lorenz(t,x,Beta),tspan,x0,options);
% Generate noisy x
NSR = 0.001; % Noise-to-signal ratio
% find the eta vale
for i = 1:n
    eta = sqrt(NSR*median(x(:,i).^2));
    nx(:,i) = x(:,i) + eta*randn(size(x,1),1);
end
% Filter noise                
T = dt;             % Sampling period       
L = size(x,1);             % Length of signal
t = (0:L-1)*T;        % Time vector
fl = 1:0.1:10;
Fs = 1/dt;
for dim = 1:3
    % find optimal cut-off frequency
    RMSE = [];
    for k = 1:length(fl)
        [b,a] = butter(5,fl(k)/(Fs/2),'low');
        clean_x_tmp = filtfilt(b,a,nx(:,dim));
        RMSE(k) = sqrt(mean((clean_x_tmp-x(:,dim)).^2));
    end

    [~,idx] = min(RMSE);
    [b,a] = butter(5,fl(idx)/(Fs/2),'low');
    cx(:,dim) = filtfilt(b,a,nx(:,dim));
end
% cx = x; % Noise-free
% Compute derivative using fourth order central difference
V = cx;
dV = zeros(length(V)-5,n);
for i = 3:length(V)-3
    for k = 1:n
        dV(i-2,k) = (1/(12*dt))*(-V(i+2,k)+8*V(i+1,k)-8*V(i-1,k)+V(i-2,k));
    end
end  
dx(3:length(V)-3,:) = dV;
%% Create intial dataset
Ts_init = 0.016;
% Sample k more intial data points
tnow = dt;
k = 2;
Xtmp = x0';
dXtmp = lorenz(0,Xtmp,Beta)';
for i = 1:k
    tnow = tnow + Ts_init;
    Xtmp = [Xtmp; cx(ceil(tnow/dt),:)];
    dXtmp = [dXtmp; dx(ceil(tnow/dt),:)];
end
% Generate full-period data with lowest Ts
% calculate F-norm
Theta = poolData(Xtmp,n,polyorder);
lambda = 0.1;
Xi_hat = sparsifyDynamics(Theta,dXtmp,lambda,n);
Fnorm_error = norm(abs(Xi_true)-abs(Xi_hat),'fro')^2;
% Compute mutual information
acf_x = autocorr(Xtmp(:,1));
acf_y = autocorr(Xtmp(:,2));
acf_z = autocorr(Xtmp(:,3));
MI = abs(mean([abs(acf_x(2)) abs(acf_y(2)) abs(acf_z(2))]));

state = [Fnorm_error log(cond(Theta)) log(real(trace(inv(Theta'*Theta))))  MI tnow];
%% Simulate data using random sampling strategies
% action probabilities
p = [1/3 1/3 1/3];
% p = [0 0 1]; % Uniform sampling
M = 10000; % number of simulations
encoded_seq = zeros(M,1);
% The lower limit for Ts
Ts_low = 0.004;
% The upper limit for Ts
Ts_high = 0.32;
tol = 0.01;
max_iter = 200;
total_reward = zeros(1,M);
Lambda_trace = 0.01;
Lambda_condnum = 0.01;
action_seq = [];
for k = 1:M
    % Creat inital dataset
    X = Xtmp;
    dX = dXtmp;
    tmp_seq = []; % action sequence
    Fnorm_error = 10000;
    count = 0;
    t = tnow;
    Ts = Ts_init;
    while count < max_iter
        count = count + 1;
        r = mnrnd(1,p);
        action = find(r == 1);
        tmp = (action == 3 && Ts < Ts_high)*Ts*2 + (action == 2)*Ts + ...
            (action == 1 && Ts > Ts_low)*Ts/2;
        Ts = (tmp == 0)*Ts + (tmp ~= 0)*tmp;
        t = t + Ts;
        if action == 3 && tmp ~= 0 
            tmp_seq  = [tmp_seq  3];
        elseif action == 2 || tmp == 0
            tmp_seq  = [tmp_seq  2];
        else
           tmp_seq  = [tmp_seq  1];
        end
        % perform the sampling action
        X = [X; cx(ceil(t/dt),:)];
        dX = [dX; dx(ceil(t/dt),:)];  
        Theta = poolData(X,n,polyorder);
%         total_reward(k) = total_reward(k)-Lambda_trace*real(log(real(trace(inv(Theta'*Theta)))));
        total_reward(k) = total_reward(k)-Lambda_condnum*log(cond(Theta));
    end
    action_seq = [action_seq; tmp_seq];
end
%% Sort the encoded sequence and plot
% Obtain the best sampling strategy
[max_t,idx] = max(total_reward);
best_action = action_seq(idx,:);
% Reperform the sampling strategy and obtain statistics
% Creat inital dataset
X = Xtmp;
dX = dXtmp;
t = tnow;
for k = 1:length(best_action)
    action = best_action(k);
    tmp = (action == 3 && Ts < Ts_high)*Ts*2 + (action == 2)*Ts + ...
        (action == 1 && Ts > Ts_low)*Ts/2;
    Ts = (tmp == 0)*Ts + (tmp ~= 0)*tmp;
    t = t + Ts;
    X = [X; cx(ceil(t/dt),:)];
    dX = [dX; dx(ceil(t/dt),:)];  
    Theta = poolData(X,n,polyorder);
    Xi_hat = sparsifyDynamics(Theta,dX,lambda,n);
    Fnorm_error = norm(abs(Xi_true)-abs(Xi_hat),'fro')^2;  
    acf_x = autocorr(X(:,1));
    acf_y = autocorr(X(:,2));
    acf_z = autocorr(X(:,3));
    MI = abs(mean([abs(acf_x(2)) abs(acf_y(2)) abs(acf_z(2))]));
    
    state(end+1,:) = [Fnorm_error log(cond(Theta)) real(log(real(trace(inv(Theta'*Theta))))) MI t];
end
% Plot the statistics
state = real(state);
figure('Position',[680,50,976,946])
subplot(4,1,1)
semilogy(state(:,1))
ylabel('Fnorm error')
grid on
set(gca,'FontSize',14','FontWeight','bold')
xlim([0 max_iter])

subplot(4,1,2)
plot(state(:,2));
grid on
ylabel('Log(\kappa)')
set(gca,'FontSize',14','FontWeight','bold')
xlim([0 max_iter])

subplot(4,1,3)
plot(state(:,3))
grid on
ylabel('Log(tr(I))')
set(gca,'FontSize',14','FontWeight','bold')
xlim([0 max_iter])

subplot(4,1,4)
plot(state(:,4))
ylabel('MI')
grid on
xlabel('Training step')
set(gca,'FontSize',14','FontWeight','bold')
xlim([0 max_iter])