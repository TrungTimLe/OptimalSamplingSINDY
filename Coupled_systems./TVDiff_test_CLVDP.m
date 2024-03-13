clear all, close all, clc
%% Generate the data
% Initialize parameters
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
dt = 0.001;
Tmax = 10;
tspan = [dt:dt:Tmax];
N = length(tspan);
par_vec = [mu,sigma,rho,beta,tau_fast,tau_slow,c1,c2];
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[t,x] = ode45(@(t,x) coupled_vdp_lorenz(t,x,par_vec),tspan,x0,options);
%% Filter the noise
Fs = 1/dt; % Sampling frequency
% Plot the frequency spectrum
for i = 1:size(x,2)
    noisy_x(:,i) = x(:,i) + 0.015*std(x(:,i))*randn(size(x,1),1);
end
% Fourier analysis                 
T = dt;             % Sampling period       
L = size(x,1);             % Length of signal
t = (0:L-1)*T;        % Time vector
Y = fft(noisy_x);
fl = 0.1:0.1:8;
RMSE = [];
dim = 3;
for k = 1:length(fl)
    [b,a] = butter(5,fl(k)/(Fs/2),'low');
    clean_x = filtfilt(b,a,noisy_x(:,dim));
    RMSE(k) = sqrt(mean((clean_x-x(:,dim)).^2));
end

[~,idx] = min(RMSE);
[b,a] = butter(5,fl(idx)/(Fs/2),'low');
clean_x = filtfilt(b,a,noisy_x(:,dim));

figure,
subplot(3,1,1)
plot(fl,RMSE,'s-','LineWidth',1.5,'MarkerSize',2)
title(['Optimal f = ' num2str(fl(idx))],'FontSize',12)
xlabel('Cut-off frequency','FontSize',12)
ylabel('RMSE','FontSize',12)
grid on

subplot(3,1,2)
plot_idx = 200;
plot(x(1:plot_idx ,dim),'--','LineWidth',1.5)
hold on
plot(noisy_x(1:plot_idx,dim),'--','LineWidth',1)
plot(clean_x(1:plot_idx),'-','LineWidth',1.5)
legend('True x','Noisy x','Clean x');
xlabel('Time t','FontSize',12)
ylabel('x','FontSize',12)
grid on
% Compute the two-sided spectrum P2. Then compute the single-sided spectrum P1 based on P2 and the even-valued signal length L.

P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;
subplot(3,1,3)
plot(f,P1,'LineWidth',1.5) 
title('Single-Sided Amplitude Spectrum of x','FontSize',12)
hold on
xline(fl(idx),'r','LineWidth',1.5)
grid on
xlabel('f (Hz)')
ylabel('|P1(f)|')
xlim([0, 20])
%% alpha sensitivity for different noise level
eta = 0:0.01:0.03;
dim = 1;
for i=1:length(x)
    dx_true(i,:) = coupled_vdp_lorenz(0,x(i,:),par_vec);
end
for k = 1:length(eta)
    noise_level = eta(k);
    for i = 1:size(x,2)
        noisy_x(:,i) = x(:,i) + eta(k)*std(x(:,i))*randn(size(x,1),1);
    end
    level = 5;
    idx = (1:0.1:(level+1))-4;
    alpha = 10.^(idx); % Regularization parameter
    N = length(alpha);
    TVDiff_dx = zeros(length(alpha),size(x,1)+1);
    for i = 1:N
        TVDiff_dx(i,:) = TVRegDiff(noisy_x(:,dim), 5, alpha(i), [], 'small', 1e-12, 0.01, 0, 0);
        RMSE(i) = sqrt(mean((TVDiff_dx(i,1:end-1)'-dx_true(1:end,dim)).^2));
    end
    % Find the optimal value
    [M,I] = min(RMSE);
    RMSE_opt(k) = M;
    alpha_opt(k) = alpha(I);
end
subplot(2,1,1)
plot(eta,alpha_opt,'LineWidth',1.5)
xlabel('\eta'),ylabel('Optimal \alpha')
grid on

subplot(2,1,2)
plot(eta,RMSE_opt,'LineWidth',1.5)
xlabel('\eta'),ylabel('Min RMSE')
grid on
