clear all, close all, clc
%% Generate Data
n = 4;
% Initialize parameters
mu1 = 5;
mu2 = 5;
c1 = 0.005;
c2 = 1;
F = 5;
tau_fast = 0.2;
tau_slow = F*tau_fast;
par_vec = [mu1,mu2,tau_fast,tau_slow,c1,c2];
x0 = [2,0,0,2];
% Integrate
dt = 0.04;
tspan = [dt:dt:500*dt];
N = length(tspan);
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[t,x]=ode45(@(t,x) coupled_vdp(t,x,par_vec),tspan,x0,options);
% Compute derivatives
dx = zeros(length(x),n);
for i = 1:length(x)
    dx(i,:) = coupled_vdp(0,x(i,:),par_vec);
end
% Pool Data  (i.e., build library of nonlinear time series)
polyorder = 3;
Theta = poolData(x,n,polyorder);
m = size(Theta,2);

% Compute Sparse regression: sequential least squares
lambda = 0.1;      % lambda is our sparsification knob.
Xi_true = sparsifyDynamics(Theta,dx,lambda,n);
poolDataLIST({'x1','x2','x3','x4'},Xi_true,n,polyorder);
%% Simulate data using random sampling strategies
n_init = 1;
n_data = length(x);
lambda_train = 0.2;
lambda_SINDy = 0;
lambda_MI = 0;
% action probabilities
p = [1/3 1/3 1/3];
% p = [0 0 1]; % Uniform sampling
M = 10000; % number of simulations
encoded_seq = zeros(M,1);
% The lower limit for Ts
Ts_low = 0.001;
% The upper limit for Ts
Ts_high = 0.64;
eta = 0;
tol = 0.001;
% eta = 0.005;
% tol = 0.01;
% eta = 0.01;
% tol = 0.05;
action_seq = "";
total_reward = zeros(1,M);
% Create the inital data
noisy_x = x0 + eta*rand(1,n);
% Compute derivative
noisy_dx = coupled_vdp(0,x0,par_vec) + eta*rand(n,1);
for k = 1:M
    k
    tmp_seq = ''; % action sequence
    Ts = dt;
    x1 = noisy_x(1:n_init,:);
    dx1 = noisy_dx(:,1:n_init)';
    Fnorm_error = 10000;
    count = 0;
    t = 0;
    while Fnorm_error > tol && count < 250
        count = count + 1;
        r = mnrnd(1,p);
        action = find(r == 1);
        tmp = (action == 3 && Ts < Ts_high)*Ts*2 + (action == 2)*Ts + ...
            (action == 1 && Ts > Ts_low)*Ts/2;
        Ts = (tmp == 0)*Ts + (tmp ~= 0)*tmp;
        t = t + Ts;
        if action == 3 && tmp ~= 0 
            tmp_seq  = [tmp_seq  '3'];
        elseif action == 2 || tmp == 0
            tmp_seq  = [tmp_seq  '2'];
        else
           tmp_seq  = [tmp_seq  '1'];
        end
        % perform the sampling action
        tspan= [Ts:Ts:3*Ts];
        [~,x_next] = ode45(@(t,x)coupled_vdp(t,x,par_vec),tspan,x1(end,:),options);
        % Add new data points
        x1 = [x1; x_next(2,:) + eta*rand(1,n)];
        dx1 = [dx1; coupled_vdp(0,x_next(2,:),par_vec)' + eta*rand(1,n)];
        Theta = poolData(x1,n,polyorder);
        Xi_hat = sparsifyDynamics(Theta,dx1,lambda,n);
        acf_x = autocorr(x1(:,1));
        acf_y = autocorr(x1(:,2));
        acf_z = autocorr(x1(:,3));
        acf_t = autocorr(x1(:,4));
        MI = abs(mean([acf_x(2) acf_y(2) acf_z(2) acf_t(2)]));
        Fnorm_error = norm(abs(Xi_true)-abs(Xi_hat),'fro')^2;
        total_reward(k) = total_reward(k)-(lambda_SINDy*log(1+Fnorm_error)+...
            lambda_MI*MI+lambda_train*t);
    end
%     train_time(k) = length(x1)*lambda_train;
    action_seq = [action_seq; tmp_seq];
%     encoded_seq(k) = str2double(action_seq(k+1));
end
%% Sort the encoded sequence and plot
% [sorted_seq,sorted_idx] = sort(encoded_seq);
% plot(total_reward(sorted_idx))
% xlabel('Sorted encoded sequence')
% ylabel('Training time t')
% Obtain the best sampling strategy
max_t = -100000;
for i = 1:length(total_reward)
    if total_reward(i) > max_t && strlength(action_seq(i+1)) < 250
        max_t = total_reward(i);
        best_seq = action_seq(i+1);
%         idx = i+1;
    end
end
max_t
disp(best_seq)
disp('1 = Upsampling')
disp('2 = Remain')
disp('3 = Downsampling')
% mean(total_reward)
% std(total_reward)