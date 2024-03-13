clear all, close all, clc
%% Set up the parameters
Beta = [10; 28; 8/3]; % Lorenz's parameters (chaotic)
n = 3; %number of state
x0 =[-8; 8; 27];  % Initial condition
%% Condition number of uniform sampling
eta = 0.2;
tol = 0.05;
Ts = [0.0025 0.005 0.01 0.02 0.04 0.08 0.16 0.32];
n_Ts = length(Ts);
x0 = initial_data(end,:); 
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
    idx = n_init;
    n_data = length(x);
    count = 0;
    Fnorm_error = 10000;
    Theta = poolData(noisy_x,n,polyorder);
    while (idx < 200) %length(x) && Fnorm_error(end) > tol)
        idx = idx + 1;
        count = count + 1;
        dx1 = [dx1; noisy_dx(idx,:)];
        Xi_hat = sparsifyDynamics(Theta(1:idx,:),dx1,lambda,n);
        cond_num(k,count) = cond(Theta(1:idx,:));
        Fnorm_error(count) = norm(abs(Xi_true)-abs(Xi_hat),'fro')^2;
    end
end
%% Plot condition number
figure,
subplot(3,2,1)
load('Lorenz_eta1.mat')
semilogy(cond_num')
grid on
xlabel('Convergence step')
ylabel('\kappa(\Theta)')
title('(a) \eta = 0')
legend('1','2','3','4','5','6','7','8')