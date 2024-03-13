clear all, close all, clc
%% Generate Data
polyorder = 3;
n = 5;
% Initialize parameters
mu = 5;
c1 = 0.01;
c2 = 10;
F = 5;
sigma = 10; rho = 28; beta = 8/3;
tau_fast = 0.2;
tau_slow = F*tau_fast;
x0 = [2 0 -8 8 27];
% Integrate
dt = 0.04;
tspan = [dt:dt:500*dt];
N = length(tspan);
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[t,x]=ode45(@(t,x) coupled_vdp_lorenz(t,x,mu,sigma,rho,beta,tau_fast,...
    tau_slow,c1,c2),tspan,x0,options);
% Plot generated data
figure,
subplot(5,1,1)
plot(tspan,x(:,1))
title('Clean data')
ylabel('x_1')
subplot(5,1,2)
plot(tspan,x(:,2))
ylabel('x_2')
subplot(5,1,3)
plot(tspan,x(:,3))
ylabel('x_3')
subplot(5,1,4)
plot(tspan,x(:,4))
ylabel('x_4')
subplot(5,1,5)
plot(tspan,x(:,5))
ylabel('x_5')
xlabel('Time')
% Generate noisy data
noise_level = 0;
for i = 1:n
    for j = 1:N
        noisy_x(j,i) = x(j,i) + noise_level*rand();
    end
end

% Plot noisy data
figure,
subplot(5,1,1)
plot(tspan,noisy_x(:,1),'r')
ylabel('x_1')
title('Noisy data')
subplot(5,1,2)
plot(tspan,noisy_x(:,2),'r')
ylabel('x_2')
subplot(5,1,3)
plot(tspan,noisy_x(:,3),'r')
ylabel('x_3')
subplot(5,1,4)
plot(tspan,noisy_x(:,4),'r')
ylabel('x_4')
subplot(5,1,5)
plot(tspan,noisy_x(:,5),'r')
ylabel('x_5')
xlabel('Time')
%% Compute derivatives
dx = zeros(length(x),n);
for i = 1:length(x)
    dx(i,:) = coupled_vdp_lorenz(0,x(i,:),mu,sigma,rho,beta,tau_fast,...
    tau_slow,c1,c2);
end
% compute noisy dx
for i = 1:length(x)
    dx_noisy(i,:) = dx(i,:);
end
%% Pool Data  (i.e., build library of nonlinear time series)
Theta = poolData(x,n,polyorder);
m = size(Theta,2);

%% Compute Sparse regression: sequential least squares
lambda = 0.4;      % lambda is our sparsification knob.
Xi_true = sparsifyDynamics(Theta,dx,lambda,n);
poolDataLIST({'x1','x2','x3','x4','x5'},Xi_true,n,polyorder);
%% Simulate error
n_init = 4;
dx1 = dx(1:n_init,:);
Theta = poolData(noisy_x,n,polyorder);
idx = n_init;
n_data = length(x);
count = 0;
Fnorm_error = 10000;
while (idx < length(x) && Fnorm_error(end) > 0.1)
    idx = idx + 1;
    count = count + 1;
    dx1 = [dx1; dx_noisy(idx,:)];
    Xi_hat = sparsifyDynamics(Theta(1:idx,:),dx1,lambda,n);
    Fnorm_error(count) = norm(abs(Xi_true)-abs(Xi_hat),'fro')^2;
end
poolDataLIST({'x1','x2','x3','x4','x5'},Xi_hat,n,polyorder);
figure,
plot(Fnorm_error)
% ylim([0 1000])
xlabel('Time t to which data are sampled up')
ylabel('Frobenius-norm error')
