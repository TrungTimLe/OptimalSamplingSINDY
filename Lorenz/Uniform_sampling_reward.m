clear all, close all, clc
%% Find the true Xi
Beta = [10; 28; 8/3]; % Lorenz's parameters (chaotic)
n = 3; %number of state
x0=[-8; 8; 27];  % Initial condition
dt = 0.04;
tspan = [dt:dt:20];
N = length(tspan);
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[t,x]=ode45(@(t,x) lorenz(t,x,Beta),tspan,x0,options);

% Compute Derivative
for i=1:length(x)
    noisy_dx(i,:) = lorenz(0,x(i,:),Beta);
end

% Pool Data  (i.e., build library of nonlinear time series)
polyorder = 3;
Theta = poolData(x,n,polyorder);
m = size(Theta,2);
% Compute Sparse regression: sequential least squares
lambda = 0.1;      % lambda is our sparsification knob.
Xi_true = sparsifyDynamics(Theta,noisy_dx,lambda,n);
poolDataLIST({'x','y','z'},Xi_true,n,polyorder);
n_init = 5;
initial_data = x(1:n_init,:);
initial_data = [];
%% Uniform sampling reward generation
lambda_train = 0.2;
lambda_SINDy = 0;
lambda_MI = 0;
eta = 0.05;
tol = 0.01;
Ts = [0.0025 0.005 0.01 0.02 0.04 0.08 0.16 0.32];
n_Ts = length(Ts);
total_reward = zeros(1,n_Ts);
% x0 = initial_data(end,:); 
x0=[-8; 8; 27];
for k = 1:n_Ts
    Ts_init = Ts(k);
    tspan=[Ts_init:Ts_init:500*Ts_init];
    options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
    [~,x]=ode45(@(t,x) lorenz(t,x,Beta),tspan,x0,options);
    x = [initial_data; x];
    N = length(x);
    noisy_x = zeros(N,n);
    for j = 1:N
        for i = 1:n
            noisy_x(j,i) = x(j,i) + eta*rand();
        end
    end
    % Compute derivative
    for i=1:N
        noisy_dx(i,:) = lorenz(0,x(i,:),Beta)+ eta*rand(3,1);
    end
    % Simulate error
    dx1 = noisy_dx(1:n_init,:);
    idx = 0;
    n_data = length(x);
    count = 0;
    Fnorm_error = 10000;
    Theta = poolData(x,n,polyorder);
    while (idx < length(x) && Fnorm_error(end) > tol)
        idx = idx + 1;
        count = count + 1;
        dx1 = [dx1; noisy_dx(idx,:)];
        Xi_hat = sparsifyDynamics(Theta(1:idx,:),dx1,lambda,n);
        Fnorm_error(count) = norm(abs(Xi_true)-abs(Xi_hat),'fro')^2;
        acf_x = autocorr(x(1:idx,1));
        acf_y = autocorr(x(1:idx,2));
        acf_z = autocorr(x(1:idx,3));
        MI = abs(mean([acf_x(2) acf_y(2) acf_z(2)]));
        total_reward(k) = total_reward(k)-(lambda_SINDy*Fnorm_error(count)+...
            lambda_MI+lambda_train*idx*Ts_init);
    end
end
disp(total_reward)
tmp = (total_reward)';
plot(Fnorm_error)