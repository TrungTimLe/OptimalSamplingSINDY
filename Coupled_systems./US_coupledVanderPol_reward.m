clear all, close all, clc
%% Find the true Xi
n = 4;
% Initialize parameters
mu1 = 5;
mu2 = 5;
c1 = 0.005;
c2 = 1;
F = 5;
tau_fast = 0.2;
tau_slow = F*tau_fast;
x0 = [2,0,0,2];
% Integrate
dt = 0.04;
tspan = [dt:dt:500*dt];
N = length(tspan);
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[t,x]=ode45(@(t,x) coupled_vdp(t,x,mu1,mu2,tau_fast,tau_slow,c1,c2),tspan,x0,options);
% Compute derivatives
dx = zeros(length(x),n);
for i = 1:length(x)
    dx(i,:) = coupled_vdp(0,x(i,:),mu1,mu2,tau_fast,tau_slow,c1,c2);
end
% Pool Data  (i.e., build library of nonlinear time series)
polyorder = 3;
Theta = poolData(x,n,polyorder);
m = size(Theta,2);

% Compute Sparse regression: sequential least squares
lambda = 0.1;      % lambda is our sparsification knob.
Xi_true = sparsifyDynamics(Theta,dx,lambda,n);
poolDataLIST({'x1','x2','x3','x4'},Xi_true,n,polyorder);
%% Uniform sampling reward generation
lambda_train = 0.1;
lambda_SINDy = 0;
lambda_MI = 0;
eta = 0.2;
Ts = [0.0025 0.005 0.01 0.02 0.04 0.08 0.16 0.32 0.64 1.28];
n_Ts = length(Ts);
total_reward = zeros(1,n_Ts);
for k = 1:n_Ts
    Ts_init = Ts(k);
    % Create the inital data
    tspan=[Ts_init:Ts_init:40];
    options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
    [t,x]=ode45(@(t,x) coupled_vdp(t,x,mu1,mu2,tau_fast,tau_slow,c1,c2),...
        tspan,x0,options);
    N = length(x);
    noisy_x = zeros(N,n);
    for j = 1:N
        for i = 1:n
            noisy_x(j,i) = x(j,i) + eta*rand();
        end
    end
    % Compute derivative
    for i=1:N
        noisy_dx(i,:) = coupled_vdp(0,x(i,:),mu1,mu2,tau_fast,tau_slow,...
            c1,c2)+ eta*rand(4,1);
    end
    % Simulate error
    n_init = 5;
    dx1 = noisy_dx(1:n_init,:);
    idx = n_init;
    n_data = length(x);
    count = 0;
    Fnorm_error = 10000;
    Theta = poolData(noisy_x,n,polyorder);
    while (idx < length(x) && Fnorm_error(end) > 40)
        idx = idx + 1;
        count = count + 1;
        dx1 = [dx1; noisy_dx(idx,:)];
        Xi_hat = sparsifyDynamics(Theta(1:idx,:),dx1,lambda,n);
        Fnorm_error(count) = norm(abs(Xi_true)-abs(Xi_hat),'fro')^2;
        acf_x = autocorr(x(1:idx,1));
        acf_y = autocorr(x(1:idx,2));
        acf_z = autocorr(x(1:idx,3));
        acf_t = autocorr(x(1:idx,4));
        MI = abs(mean([acf_x(2) acf_y(2) acf_z(2) acf_t(2)]));
        total_reward(k) = total_reward(k)-(lambda_SINDy*Fnorm_error(count)+...
            lambda_MI*MI-lambda_train*(count*Ts_init));
    end
end
disp(total_reward+1)
tmp = (total_reward+1)';
plot(Fnorm_error)