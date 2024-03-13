clear all, close all, clc
% %% Find the true Xi
% Beta = [10; 28; 8/3]; % Lorenz's parameters (chaotic)
% n = 3; %number of state
% x0 =[-8; 8; 27];  % Initial condition
% dt = 0.04;
% tspan=[dt:dt:20];
% N = length(tspan);
% options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
% [t,x]=ode45(@(t,x) lorenz(t,x,Beta),tspan,x0,options);
% for i=1:length(x)
%     noisy_dx(i,:) = lorenz(0,x(i,:),Beta);
% end
% 
% % Pool Data  (i.e., build library of nonlinear time series)
% polyorder = 3;
% Theta = poolData(x,n,polyorder);
% m = size(Theta,2);
% % Compute Sparse regression: sequential least squares
% lambda = 0.1;      % lambda is our sparsification knob.
% Xi_true = sparsifyDynamics(Theta,noisy_dx,lambda,n);
% poolDataLIST({'x','y','z'},Xi_true,n,polyorder);
% n_init = 1;
% initial_data = x(1:n_init,:);
% % initial_data = [];
% %% Condition number of uniform sampling
% eta = 0.2;
% tol = 0.05;
% Ts = [0.0025 0.005 0.01 0.02 0.04 0.08 0.16 0.32];
% n_Ts = length(Ts);
% x0 = initial_data(end,:); 
% x0=[-8; 8; 27];
% for k = 1:n_Ts
%     Ts_init = Ts(k);
%     tspan=[Ts_init:Ts_init:500*Ts_init];
%     options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
%     [~,x]=ode45(@(t,x) lorenz(t,x,Beta),tspan,x0,options);
%     x = [initial_data; x];
%     N = length(x);
%     noisy_x = zeros(N,n);
%     for j = 1:N
%         for i = 1:n
%             noisy_x(j,i) = x(j,i) + eta*rand();
%         end
%     end
%     % Compute derivative
%     for i=1:N
%         noisy_dx(i,:) = lorenz(0,x(i,:),Beta)+ eta*rand(3,1);
%     end
%     % Simulate error
%     dx1 = noisy_dx(1:n_init,:);
%     idx = n_init;
%     n_data = length(x);
%     count = 0;
%     Fnorm_error = 10000;
%     Theta = poolData(noisy_x,n,polyorder);
%     while (idx < 200) %length(x) && Fnorm_error(end) > tol)
%         idx = idx + 1;
%         count = count + 1;
%         dx1 = [dx1; noisy_dx(idx,:)];
%         Xi_hat = sparsifyDynamics(Theta(1:idx,:),dx1,lambda,n);
%         cond_num(k,count) = cond(Theta(1:idx,:));
%         Fnorm_error(count) = norm(abs(Xi_true)-abs(Xi_hat),'fro')^2;
%     end
% end

%% Condition number for RL 
%% Case 1: Only training time
% eta = 0
polyorder = 3;
N = 1000;
Tmax = 10;
Beta = [10; 28; 8/3]; % Lorenz's parameters (chaotic)
n = 3; %number of state
x0 = [-8; 8; 27]';  % Initial condition
RL_strat =  [-1 -1 -1 -1 0 0 1 0 0 0 0 0 0 0 -1 -1 -1 -1 -1];
best_strat = [1 1 1 1 1 1 3 1 2 2 2 2 3 1 2 2 3 3 2];
% Generate the data
% Integrate
dt = 0.0025*0.25;
tspan = [dt:dt:Tmax];
N = length(tspan);
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[t,x]=ode45(@(t,x) lorenz(t,x,Beta),tspan,x0,options);
L = 1:length(x);

Ts = 0.04;
idx = 1;
Ds = zeros(length(RL_strat),n); % Initialize Ds
Ts_low = 0.001;
Ts_high = 0.64;
count = 0;
for i = 1:length(RL_strat)
    count = count + 1;
    tmp = RL_strat(i);
    Ts = 0.5*Ts*(tmp == -1 && Ts > Ts_low) + Ts*(tmp == 0 || (tmp == -1 && Ts <= Ts_low) ...
        || (tmp == 1 && Ts >= Ts_high)) + 2*Ts*(tmp == 1 && Ts < Ts_high);
    idx = idx + Ts/dt;
    Ds(i,:) = x(idx,:);
    Theta = poolData([x0; Ds(1:i,:)],n,polyorder);
    RL_con_num(1,count) = cond(Theta);
end

Ts = 0.04;
idx = 1;
Ds = zeros(length(best_strat),n); % Initialize Ds
count = 0;
for i = 1:length(best_strat)
    count = count + 1;
    tmp = best_strat(i);
    Ts = 0.5*Ts*(tmp == 1 && Ts > Ts_low) + Ts*(tmp == 2 || (tmp == 1 && Ts <= Ts_low) ...
        || (tmp == 3 && Ts >= Ts_high)) + 2*Ts*(tmp == 3 && Ts < Ts_high);
    idx = idx + Ts/dt;
    Ds(i,:) = x(idx,:);
    Theta = poolData([x0; Ds(1:i,:)],n,polyorder);
    Best_con_num(1,count) = cond(Theta);
end

% eta = 0.05
% Generate noisy data
eta = 0.05;
x1 = x + eta*rand(length(x),n);
RL_strat =  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 -1 1];
best_strat = [1 1 3 2 1 3 2 3 1 3 2 2 1 3 1 2 1 1];

Ts = 0.04;
% Plot RL_strat
idx = 1;
Ds = zeros(length(RL_strat),n); % Initialize Ds
Ts_low = 0.001;
Ts_high = 0.64;
count = 0;
for i = 1:length(RL_strat)
    count = count + 1;
    tmp = RL_strat(i);
    Ts = 0.5*Ts*(tmp == -1 && Ts > Ts_low) + Ts*(tmp == 0 || (tmp == -1 && Ts <= Ts_low) ...
        || (tmp == 1 && Ts >= Ts_high)) + 2*Ts*(tmp == 1 && Ts < Ts_high);
    idx = idx + Ts/dt;
    Ds(i,:) = x(idx,:);
    Theta = poolData([x0; Ds(1:i,:)],n,polyorder);
    RL_con_num(2,count) = cond(Theta);
end

Ts = 0.04;
idx = 1;
Ds = zeros(length(best_strat),n); % Initialize Ds
count = 0;
for i = 1:length(best_strat)
    count = count + 1;
    tmp = best_strat(i);
    Ts = 0.5*Ts*(tmp == 1 && Ts > Ts_low) + Ts*(tmp == 2 || (tmp == 1 && Ts <= Ts_low) ...
        || (tmp == 3 && Ts >= Ts_high)) + 2*Ts*(tmp == 3 && Ts < Ts_high);
    idx = idx + Ts/dt;
    Ds(i,:) = x(idx,:);
    Theta = poolData([x0; Ds(1:i,:)],n,polyorder);
    Best_con_num(2,count) = cond(Theta);
end

% eta = 0.2
eta = 0.2;
x1 = x + eta*rand(length(x),n);
RL_strat =  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
best_strat = [1 3 2 1 3 1 3 1 2 2 3 3 1 1 3 1 3 2];
% Plot the attractor
% Generate the data
% Integrate
Ts = 0.04;
% Plot RL_strat
idx = 1;
Ds = zeros(length(RL_strat),n); % Initialize Ds
Ts_low = 0.001;
Ts_high = 0.64;
count = 0;
for i = 1:length(RL_strat)
    count = count + 1;
    tmp = RL_strat(i);
    Ts = 0.5*Ts*(tmp == -1 && Ts > Ts_low) + Ts*(tmp == 0 || (tmp == -1 && Ts <= Ts_low) ...
        || (tmp == 1 && Ts >= Ts_high)) + 2*Ts*(tmp == 1 && Ts < Ts_high);
    idx = idx + Ts/dt;
    Ds(i,:) = x(idx,:);
    Theta = poolData([x0; Ds(1:i,:)],n,polyorder);
    RL_con_num(3,count) = cond(Theta);
end

Ts = 0.04;
idx = 1;
Ds = zeros(length(best_strat),n); % Initialize Ds
count = 0;
for i = 1:length(best_strat)
    count = count + 1;
    tmp = best_strat(i);
    Ts = 0.5*Ts*(tmp == 1 && Ts > Ts_low) + Ts*(tmp == 2 || (tmp == 1 && Ts <= Ts_low) ...
        || (tmp == 3 && Ts >= Ts_high)) + 2*Ts*(tmp == 3 && Ts < Ts_high);
    idx = idx + Ts/dt;
    Ds(i,:) = x(idx,:);
    Theta = poolData([x0; Ds(1:i,:)],n,polyorder);
    Best_con_num(3,count) = cond(Theta);
end
%% Case 2: Only SINDY loss
% eta = 0
RL_con_num = [];
Best_con_num = [];
Beta = [10; 28; 8/3]; % Lorenz's parameters (chaotic)
n = 3; %number of state
RL_strat =  [1 1 1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 1];
best_strat = [2 2 3 2 2 3 2 2 1 2 3 3 3 2 2 2 2];
% Plot the attractor
% Generate the data
% Integrate
dt = 0.0025*0.25;
tspan = [dt:dt:Tmax];
N = length(tspan);
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[t,x]=ode45(@(t,x) lorenz(t,x,Beta),tspan,x0,options);
L = 1:length(x);

Ts = 0.04;
% Plot RL_strat
idx = 1;
Ds = zeros(length(RL_strat),n); % Initialize Ds
Ts_low = 0.001;
Ts_high = 0.64;
count = 0;
for i = 1:length(RL_strat)
    count = count + 1;
    tmp = RL_strat(i);
    Ts = 0.5*Ts*(tmp == -1 && Ts > Ts_low) + Ts*(tmp == 0 || (tmp == -1 && Ts <= Ts_low) ...
        || (tmp == 1 && Ts >= Ts_high)) + 2*Ts*(tmp == 1 && Ts < Ts_high);
    idx = idx + Ts/dt;
    Ds(i,:) = x(idx,:);
    Theta = poolData([x0; Ds(1:i,:)],n,polyorder);
    RL_con_num(1,count) = cond(Theta);
end

Ts = 0.04;
idx = 1;
Ds = zeros(length(best_strat),n); % Initialize Ds
count = 0;
for i = 1:length(best_strat)
    count = count + 1;
    tmp = best_strat(i);
    Ts = 0.5*Ts*(tmp == 1 && Ts > Ts_low) + Ts*(tmp == 2 || (tmp == 1 && Ts <= Ts_low) ...
        || (tmp == 3 && Ts >= Ts_high)) + 2*Ts*(tmp == 3 && Ts < Ts_high);
    idx = idx + Ts/dt;
    Ds(i,:) = x(idx,:);
    Theta = poolData([x0; Ds(1:i,:)],n,polyorder);
    Best_con_num(1,count) = cond(Theta);
end

% eta = 0.05
% Generate noisy data
eta = 0.05;
x1 = x + eta*rand(length(x),n);
RL_strat =  [1 1 1 1 1 0 0 1 1 1 1 0 0 1 0 1 1];
best_strat = [1 3 3 3 3 2 3 2 2 1 3 2 1 1 1 3 2];
% Plot the attractor

Ts = 0.04;
% Plot RL_strat
idx = 1;
Ds = zeros(length(RL_strat),n); % Initialize Ds
Ts_low = 0.001;
Ts_high = 0.64;
count = 0;
for i = 1:length(RL_strat)
    count = count + 1;
    tmp = RL_strat(i);
    Ts = 0.5*Ts*(tmp == -1 && Ts > Ts_low) + Ts*(tmp == 0 || (tmp == -1 && Ts <= Ts_low) ...
        || (tmp == 1 && Ts >= Ts_high)) + 2*Ts*(tmp == 1 && Ts < Ts_high);
    idx = idx + Ts/dt;
    Ds(i,:) = x(idx,:);
    Theta = poolData([x0; Ds(1:i,:)],n,polyorder);
    RL_con_num(2,count) = cond(Theta);
end

Ts = 0.04;
idx = 1;
Ds = zeros(length(best_strat),n); % Initialize Ds
count = 0;
for i = 1:length(best_strat)
    count = count + 1;
    tmp = best_strat(i);
    Ts = 0.5*Ts*(tmp == 1 && Ts > Ts_low) + Ts*(tmp == 2 || (tmp == 1 && Ts <= Ts_low) ...
        || (tmp == 3 && Ts >= Ts_high)) + 2*Ts*(tmp == 3 && Ts < Ts_high);
    idx = idx + Ts/dt;
    Ds(i,:) = x(idx,:);
    Theta = poolData([x0; Ds(1:i,:)],n,polyorder);
    Best_con_num(2,count) = cond(Theta);
end

% eta = 0.2
eta = 0.2;
x1 = x + eta*rand(length(x),n);
RL_strat =  [1 1 1 1 -1 1 -1 1 -1 1 1 1 -1 1 1 -1 -1 1];
best_strat = [2 2 2 3 1 2 2 2 3 3 3 2 1 1 2 1 3];
% Plot the attractor

Ts = 0.04;
% Plot RL_strat
idx = 1;
Ds = zeros(length(RL_strat),n); % Initialize Ds
Ts_low = 0.001;
Ts_high = 0.64;
count = 0;
for i = 1:length(RL_strat)
    count = count + 1;
    tmp = RL_strat(i);
    Ts = 0.5*Ts*(tmp == -1 && Ts > Ts_low) + Ts*(tmp == 0 || (tmp == -1 && Ts <= Ts_low) ...
        || (tmp == 1 && Ts >= Ts_high)) + 2*Ts*(tmp == 1 && Ts < Ts_high);
    idx = idx + Ts/dt;
    Ds(i,:) = x1(idx,:);
    Theta = poolData([x0; Ds(1:i,:)],n,polyorder);
    RL_con_num(3,count) = cond(Theta);
end

Ts = 0.04;
idx = 1;
Ds = zeros(length(best_strat),n); % Initialize Ds
count = 0;
for i = 1:length(best_strat)
    count = count + 1;
    tmp = best_strat(i);
    Ts = 0.5*Ts*(tmp == 1 && Ts > Ts_low) + Ts*(tmp == 2 || (tmp == 1 && Ts <= Ts_low) ...
        || (tmp == 3 && Ts >= Ts_high)) + 2*Ts*(tmp == 3 && Ts < Ts_high);
    idx = idx + Ts/dt;
    Ds(i,:) = x(idx,:);
    Theta = poolData([x0; Ds(1:i,:)],n,polyorder);
    Best_con_num(3,count) = cond(Theta);
end
%% Case 3: Only MI
% eta = 0
RL_con_num = [];
Best_con_num = [];
Beta = [10; 28; 8/3]; % Lorenz's parameters (chaotic)
n = 3; %number of state
RL_strat =  [1 1 1 1 1 1 0 0 0 0 -1 1 0 0 0 1 1 1];
best_strat = [3 1 3 3 3 3 2 1 2 2 3 2 1 2 2 2 1 2];
% Plot the attractor
% Generate the data
% Integrate
dt = 0.0025*0.25;
tspan = [dt:dt:Tmax];
N = length(tspan);
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[t,x]=ode45(@(t,x) lorenz(t,x,Beta),tspan,x0,options);
L = 1:length(x);

Ts = 0.04;
% Plot RL_strat
idx = 1;
Ds = zeros(length(RL_strat),n); % Initialize Ds
Ts_low = 0.001;
Ts_high = 0.64;
count = 0;
for i = 1:length(RL_strat)
    count = count + 1;
    tmp = RL_strat(i);
    Ts = 0.5*Ts*(tmp == -1 && Ts > Ts_low) + Ts*(tmp == 0 || (tmp == -1 && Ts <= Ts_low) ...
        || (tmp == 1 && Ts >= Ts_high)) + 2*Ts*(tmp == 1 && Ts < Ts_high);
    idx = idx + Ts/dt;
    Ds(i,:) = x(idx,:);
    Theta = poolData([x0; Ds(1:i,:)],n,polyorder);
    RL_con_num(1,count) = cond(Theta);
end

Ts = 0.04;
idx = 1;
Ds = zeros(length(best_strat),n); % Initialize Ds
count = 0;
for i = 1:length(best_strat)
    count = count + 1;
    tmp = best_strat(i);
    Ts = 0.5*Ts*(tmp == 1 && Ts > Ts_low) + Ts*(tmp == 2 || (tmp == 1 && Ts <= Ts_low) ...
        || (tmp == 3 && Ts >= Ts_high)) + 2*Ts*(tmp == 3 && Ts < Ts_high);
    idx = idx + Ts/dt;
    Ds(i,:) = x(idx,:);
    Theta = poolData([x0; Ds(1:i,:)],n,polyorder);
    Best_con_num(1,count) = cond(Theta);
end

% eta = 0.05
% Generate noisy data
eta = 0.05;
x1 = x + eta*rand(length(x),n);
RL_strat =  [1 1 1 1 1 1 0 1 1 1 1 1 0 1 0 1 1];
best_strat = [2 3 2 3 3 3 2 2 2 2 1 1 2 1 1 1 3 3];
% Plot the attractor

Ts = 0.04;
% Plot RL_strat
idx = 1;
Ds = zeros(length(RL_strat),n); % Initialize Ds
Ts_low = 0.001;
Ts_high = 0.64;
count = 0;
for i = 1:length(RL_strat)
    count = count + 1;
    tmp = RL_strat(i);
    Ts = 0.5*Ts*(tmp == -1 && Ts > Ts_low) + Ts*(tmp == 0 || (tmp == -1 && Ts <= Ts_low) ...
        || (tmp == 1 && Ts >= Ts_high)) + 2*Ts*(tmp == 1 && Ts < Ts_high);
    idx = idx + Ts/dt;
    Ds(i,:) = x1(idx,:);
    Theta = poolData([x0; Ds(1:i,:)],n,polyorder);
    RL_con_num(2,count) = cond(Theta);
end

Ts = 0.04;
idx = 1;
Ds = zeros(length(best_strat),n); % Initialize Ds
count = 0;
for i = 1:length(best_strat)
    count = count + 1;
    tmp = best_strat(i);
    Ts = 0.5*Ts*(tmp == 1 && Ts > Ts_low) + Ts*(tmp == 2 || (tmp == 1 && Ts <= Ts_low) ...
        || (tmp == 3 && Ts >= Ts_high)) + 2*Ts*(tmp == 3 && Ts < Ts_high);
    idx = idx + Ts/dt;
    Ds(i,:) = x1(idx,:);
    Theta = poolData([x0; Ds(1:i,:)],n,polyorder);
    Best_con_num(2,count) = cond(Theta);
end

% eta = 0.2
eta = 0.2;
x1 = x + eta*rand(length(x),n);
RL_strat =  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
best_strat = [3 2 3 3 3 2 2 2 1 1 1 3 1 3 2 1 1 3];
% Plot the attractor

Ts = 0.04;
% Plot RL_strat
idx = 1;
Ds = zeros(length(RL_strat),n); % Initialize Ds
Ts_low = 0.001;
Ts_high = 0.64;
count = 0;
for i = 1:length(RL_strat)
    count = count + 1;
    tmp = RL_strat(i);
    Ts = 0.5*Ts*(tmp == -1 && Ts > Ts_low) + Ts*(tmp == 0 || (tmp == -1 && Ts <= Ts_low) ...
        || (tmp == 1 && Ts >= Ts_high)) + 2*Ts*(tmp == 1 && Ts < Ts_high);
    idx = idx + Ts/dt;
    Ds(i,:) = x1(idx,:);
    Theta = poolData([x0; Ds(1:i,:)],n,polyorder);
    RL_con_num(3,count) = cond(Theta);
end

Ts = 0.04;
idx = 1;
Ds = zeros(length(best_strat),n); % Initialize Ds
count = 0;
for i = 1:length(best_strat)
    count = count + 1;
    tmp = best_strat(i);
    Ts = 0.5*Ts*(tmp == 1 && Ts > Ts_low) + Ts*(tmp == 2 || (tmp == 1 && Ts <= Ts_low) ...
        || (tmp == 3 && Ts >= Ts_high)) + 2*Ts*(tmp == 3 && Ts < Ts_high);
    idx = idx + Ts/dt;
    Ds(i,:) = x1(idx,:);
    Theta = poolData([x0; Ds(1:i,:)],n,polyorder);
    Best_con_num(3,count) = cond(Theta);
end
%% Case 4: Combined signal
% eta = 0
RL_con_num = [];
Best_con_num = [];
n = 3; %number of state
RL_strat =  [0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1];
best_strat = [2 1 1 1 1 1 1 3 1 2 2 2 2 2 3 1 2 2 3];
% Plot the attractor
% Generate the data
% Integrate
dt = 0.0025*0.25;
tspan = [dt:dt:Tmax];
N = length(tspan);
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[t,x]=ode45(@(t,x) lorenz(t,x,Beta),tspan,x0,options);

Ts = 0.04;
% Plot RL_strat
idx = 1;
Ds = zeros(length(RL_strat),n); % Initialize Ds
Ts_low = 0.001;
Ts_high = 0.64;
count = 0;
for i = 1:length(RL_strat)
    count = count + 1;
    tmp = RL_strat(i);
    Ts = 0.5*Ts*(tmp == -1 && Ts > Ts_low) + Ts*(tmp == 0 || (tmp == -1 && Ts <= Ts_low) ...
        || (tmp == 1 && Ts >= Ts_high)) + 2*Ts*(tmp == 1 && Ts < Ts_high);
    idx = idx + Ts/dt;
    Ds(i,:) = x(idx,:);
    Theta = poolData([x0; Ds(1:i,:)],n,polyorder);
    RL_con_num(1,count) = cond(Theta);
end

Ts = 0.04;
idx = 1;
Ds = zeros(length(best_strat),n); % Initialize Ds
count = 0;
for i = 1:length(best_strat)
    count = count + 1;
    tmp = best_strat(i);
    Ts = 0.5*Ts*(tmp == 1 && Ts > Ts_low) + Ts*(tmp == 2 || (tmp == 1 && Ts <= Ts_low) ...
        || (tmp == 3 && Ts >= Ts_high)) + 2*Ts*(tmp == 3 && Ts < Ts_high);
    idx = idx + Ts/dt;
    Ds(i,:) = x(idx,:);
    Theta = poolData([x0; Ds(1:i,:)],n,polyorder);
    Best_con_num(1,count) = cond(Theta);
end

% eta = 0.05
% Generate noisy data
eta = 0.05;
x1 = x + eta*rand(length(x),n);
RL_strat =  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
best_strat = [2 1 1 3 2 2 2 3 2 2 1 3 2 3 1 3 2 2];
% Plot the attractor


Ts = 0.04;
% Plot RL_strat
idx = 1;
Ds = zeros(length(RL_strat),n); % Initialize Ds
Ts_low = 0.001;
Ts_high = 0.64;
count = 0;
for i = 1:length(RL_strat)
    count = count + 1;
    tmp = RL_strat(i);
    Ts = 0.5*Ts*(tmp == -1 && Ts > Ts_low) + Ts*(tmp == 0 || (tmp == -1 && Ts <= Ts_low) ...
        || (tmp == 1 && Ts >= Ts_high)) + 2*Ts*(tmp == 1 && Ts < Ts_high);
    idx = idx + Ts/dt;
    Ds(i,:) = x1(idx,:);
    Theta = poolData([x0; Ds(1:i,:)],n,polyorder);
    RL_con_num(2,count) = cond(Theta);
end

Ts = 0.04;
idx = 1;
Ds = zeros(length(best_strat),n); % Initialize Ds
count = 0;
for i = 1:length(best_strat)
    count = count + 1;
    tmp = best_strat(i);
    Ts = 0.5*Ts*(tmp == 1 && Ts > Ts_low) + Ts*(tmp == 2 || (tmp == 1 && Ts <= Ts_low) ...
        || (tmp == 3 && Ts >= Ts_high)) + 2*Ts*(tmp == 3 && Ts < Ts_high);
    idx = idx + Ts/dt;
    Ds(i,:) = x1(idx,:);
    Theta = poolData([x0; Ds(1:i,:)],n,polyorder);
    Best_con_num(2,count) = cond(Theta);
end

% eta = 0.2
eta = 0.2;
x1 = x + eta*rand(length(x),n);
RL_strat =  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
best_strat = [1 1 1 3 3 1 2 3 3 3 2 2 2 1 2 2 3 1];

Ts = 0.04;
% Plot RL_strat
idx = 1;
Ds = zeros(length(RL_strat),n); % Initialize Ds
Ts_low = 0.001;
Ts_high = 0.64;
count = 0;
for i = 1:length(RL_strat)
    count = count + 1;
    tmp = RL_strat(i);
    Ts = 0.5*Ts*(tmp == -1 && Ts > Ts_low) + Ts*(tmp == 0 || (tmp == -1 && Ts <= Ts_low) ...
        || (tmp == 1 && Ts >= Ts_high)) + 2*Ts*(tmp == 1 && Ts < Ts_high);
    idx = idx + Ts/dt;
    Ds(i,:) = x1(idx,:);
    Theta = poolData([x0; Ds(1:i,:)],n,polyorder);
    RL_con_num(3,count) = cond(Theta);
end

Ts = 0.04;
idx = 1;
Ds = zeros(length(best_strat),n); % Initialize Ds
count = 0;
for i = 1:length(best_strat)
    count = count + 1;
    tmp = best_strat(i);
    Ts = 0.5*Ts*(tmp == 1 && Ts > Ts_low) + Ts*(tmp == 2 || (tmp == 1 && Ts <= Ts_low) ...
        || (tmp == 3 && Ts >= Ts_high)) + 2*Ts*(tmp == 3 && Ts < Ts_high);
    idx = idx + Ts/dt;
    Ds(i,:) = x1(idx,:);
    Theta = poolData([x0; Ds(1:i,:)],n,polyorder);
    Best_con_num(3,count) = cond(Theta);
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

subplot(3,2,2)
load('Lorenz_eta2.mat')
semilogy(cond_num')
grid on
xlabel('Convergence step')
ylabel('\kappa(\Theta)')
title('(b) \eta = 0.05')

subplot(3,2,3)
load('Lorenz_eta3.mat')
semilogy(cond_num')
grid on
xlabel('Convergence step')
ylabel('\kappa(\Theta)')
title('(c) \eta = 0.2')

subplot(3,2,4)
load('Lorenz_RL1.mat')
semilogy(RL_con_num')
grid on
hold on
load('Lorenz_best1.mat')
semilogy(Best_con_num',':')
load('Lorenz_eta1.mat')
semilogy(cond_num(5,:)','--')
load('Lorenz_eta2.mat')
semilogy(cond_num(5,:)','--')
load('Lorenz_eta3.mat')
semilogy(cond_num(5,:)','--')
xlabel('Convergence step')
ylabel('\kappa(\Theta)')
title('(d) Only training time')
legend('RL1','RL2','RL3','Best1','Best2','Best3','eta1','eta2','eta3')

subplot(3,2,5)
load('Lorenz_RL2.mat')
semilogy(RL_con_num')
grid on
hold on
load('Lorenz_best2.mat')
semilogy(Best_con_num',':')
load('Lorenz_eta1.mat')
semilogy(cond_num(5,:)','--')
load('Lorenz_eta2.mat')
semilogy(cond_num(5,:)','--')
load('Lorenz_eta3.mat')
semilogy(cond_num(5,:)','--')
xlabel('Convergence step')
ylabel('\kappa(\Theta)')
title('(e) Only SINDY loss')

subplot(3,2,6)
load('Lorenz_RL3.mat')
semilogy(RL_con_num')
grid on
hold on
load('Lorenz_best3.mat')
semilogy(Best_con_num',':')
load('Lorenz_eta1.mat')
semilogy(cond_num(5,:)','--')
load('Lorenz_eta2.mat')
semilogy(cond_num(5,:)','--')
load('Lorenz_eta3.mat')
semilogy(cond_num(5,:)','--')
xlabel('Convergence step')
ylabel('\kappa(\Theta)')
title('(f) Only MI')
