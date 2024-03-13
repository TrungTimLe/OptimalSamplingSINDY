close all, clear all, clc
%% Plot the best and RL sampling policies
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
%% Plot the data
Ts = 0.04;
Tmax = 20;

RL_strat =      [1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1
     1     1     1     1     1     1     1     1     1    -1     1     1     1     1     1     1     1     1
     1     1     1     1     1    -1    -1     1     1     1     1     1     1     1     1     1     1     1];

best_strat = [3 3 3 3 2 1 3 3 2 3 1 3 1 3 1 3 2 2 2 2 1 1 2 3 1 2 3 3 2 2 2 2 2 2 2 1 3 2 1 3 1 2 3 2 2 2 2 2 1 1 2 1 2 3 2];
% Generate the data
% Integrate
dt = 0.001;
tspan = [dt:dt:Tmax];
N = length(tspan);
par_vec = [mu,sigma,rho,beta,tau_fast,tau_slow,c1,c2];
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[t,x]=ode45(@(t,x) coupled_vdp_lorenz(t,x,par_vec),tspan,x0,options);
% generate the trajectory data
tspan = [dt:dt:30];
N = length(tspan);
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[t,x]=ode45(@(t,x) coupled_vdp_lorenz(t,x,par_vec),tspan,x0,options);
% Plot best_strat
idx = 1;
Ts_low = 0.001;
Ts_high = 0.64;
X = zeros(length(best_strat),n); % Initialize Ds
for i = 1:size(x,2)
    x1(:,i) = x(:,i) + 0.015*std(x(:,i))*randn(size(x,1),1);
end
for i = 1:length(best_strat)
    tmp = best_strat(i);
    Ts = 0.5*Ts*(tmp == 1 && Ts > Ts_low) + Ts*(tmp == 2 || (tmp == 1 && Ts <= Ts_low) ...
        || (tmp == 3 && Ts >= Ts_high)) + 2*Ts*(tmp == 3 && Ts < Ts_high);
    idx = idx + Ts/dt;
    X(i,:) = x1(idx,:);
end
%% Plot the attractor
figure('color','w','Position',[639,494,1069,484])
% Plot the attractor of fast VDP
L = 1:30000;
subplot(1,2,1)
plot(x1(L,1),x1(L,2),'Color',[.1 .1 .1],'LineWidth',0.5)
hold on
scatter(x1(1,1),x1(1,2),'filled','b')
axis on
grid on
axis tight
xlabel('x_1'), ylabel('x_2')
set(gca,'FontSize',18,'FontWeight','bold')
set(gcf,'PaperPositionMode','auto')
n_sample = 40;
plot(X(2:n_sample,1),X(2:n_sample,2),'o','MarkerSize',5,'Color',[0.75 0 0],...
    'LineWidth',0.5,'MarkerFaceColor',[0.75 0 0])
legend('Attractor','Inital point','Sampled points','FontSize',14,'Location','northwest')
title('Noisy "fast" VDP attractor','FontSize',20)
% Plot the attractor of slow Lorenz
subplot(1,2,2)
L = 1:30000;
plot3(x1(L,3),x1(L,4),x1(L,5),'Color',[.1 .1 .1],'LineWidth',0.5)
hold on
scatter3(x1(1,3),x1(1,4),x1(1,5),'filled','b')
axis on
grid on
view(-5,12)
axis tight
xlabel('x_3'), ylabel('x_4'), zlabel('x_5')
set(gca,'FontSize',18,'FontWeight','bold')
% set(gcf,'Position',[100 100 600 400])
set(gcf,'PaperPositionMode','auto')
plot3(X(2:n_sample,3),X(2:n_sample,4),X(2:n_sample,5),'o','MarkerSize',5,'Color',[0.75 0 0],...
    'LineWidth',0.5,'MarkerFaceColor',[0.75 0 0])
% legend('Lorenz attractor','Inital point','FontSize',14,'Location','north')
title('Noisy "slow" Lorenz attractor','FontSize',20)
%% Plot the time series data
for i = 1:size(x,2)
    x2(:,i) = x(:,i) + 0.15*std(x(:,i))*randn(size(x,1),1);
end
idx = 1;
for i = 1:length(best_strat)
    tmp = best_strat(i);
    Ts = 0.5*Ts*(tmp == 1 && Ts > Ts_low) + Ts*(tmp == 2 || (tmp == 1 && Ts <= Ts_low) ...
        || (tmp == 3 && Ts >= Ts_high)) + 2*Ts*(tmp == 3 && Ts < Ts_high);
    idx = idx + Ts/dt;
    X(i,:) = x2(idx,:);
    sample_idx(i+1) = idx;
end

figure('color','w','Position',[639,494,1069,484])
Tspan = 10000;
L = 1:1:Tspan;
time1 = linspace(dt,Tspan*dt,length(L));
time = linspace(dt,Tspan*dt,Tspan);
subplot(1,2,1)
plot(time1,x2(L,1),'LineWidth',1.5)
hold on
n_sample = 10;
plot(time1,x2(L,2),'LineWidth',1.5)
tmp = sample_idx(2:n_sample);
idx = find(X(2,1) == x2(:,1));
plot(time(tmp)+time(idx),X(2:n_sample,1),'o','MarkerSize',5,'Color',[0.75 0 0],...
    'LineWidth',1.5,'MarkerFaceColor',[0.75 0 0])
plot(time(tmp)+0.65,X(2:n_sample,2),'o','MarkerSize',5,'Color',[0.75 0 0],...
    'LineWidth',1.5,'MarkerFaceColor',[0.75 0 0])
axis on
grid on
axis tight
xlabel('Time t'), ylabel('Measured value')
set(gca,'FontSize',18,'FontWeight','bold')
set(gcf,'PaperPositionMode','auto')
legend('x_1','x_2','FontSize',14,'Location','northwest')
title('Noisy "fast" VDP time series','FontSize',20)

subplot(1,2,2)
plot(time1,x2(L,3),'LineWidth',1.5)
hold on
plot(time1,x2(L,4),'LineWidth',1.5)
plot(time1,x2(L,5),'LineWidth',1.5)
% plot(time(tmp)+0.7,X(2:n_sample,3),'o','MarkerSize',5,'Color',[0.75 0 0],...
%     'LineWidth',1.5,'MarkerFaceColor',[0.75 0 0])
% 
% plot(time(tmp)+0.7,X(2:n_sample,4),'o','MarkerSize',5,'Color',[0.75 0 0],...
%     'LineWidth',1.5,'MarkerFaceColor',[0.75 0 0])
% 
% plot(time(tmp)+0.7,X(2:n_sample,5),'o','MarkerSize',5,'Color',[0.75 0 0],...
%     'LineWidth',1.5,'MarkerFaceColor',[0.75 0 0])
axis on
grid on
axis tight
xlabel('Time t'), ylabel('Measured value')
set(gca,'FontSize',18,'FontWeight','bold')
set(gcf,'PaperPositionMode','auto')
legend('x_3','x_4','x_5','FontSize',14,'Location','southwest')
title('Noisy "slow" Lorenz time series','FontSize',20)
%% Filter the noise, estimate derivative and plot
dt = 0.001;
tspan = [dt:dt:10];
N = length(tspan);
par_vec = [mu,sigma,rho,beta,tau_fast,tau_slow,c1,c2];
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[t,x]=ode45(@(t,x) coupled_vdp_lorenz(t,x,par_vec),tspan,x0,options);
Fs = 1/dt; % Sampling frequency

% Fourier analysis                 
T = dt;             % Sampling period       
L = size(x,1);             % Length of signal
t = (0:L-1)*T;        % Time vector
Y = fft(x2);
for dim = 1:5
    % find optimal cut-off frequency
    fl = 3:0.1:10;
    RMSE = [];
    for k = 1:length(fl)
        [b,a] = butter(5,fl(k)/(Fs/2),'low');
        clean_x_tmp = filtfilt(b,a,x2(1:10000,dim));
        RMSE(k) = sqrt(mean((clean_x_tmp-x(:,dim)).^2));
    end

    [~,idx] = min(RMSE);
    [b,a] = butter(5,fl(idx)/(Fs/2),'low');
    clean_x(:,dim) = filtfilt(b,a,x2(:,dim));
end
% compute derivative using fourth order central difference
V = clean_x;
dV = zeros(length(V)-5,5);
for i = 3:length(V)-3
    for k = 1:5
        dV(i-2,k) = (1/(12*dt))*(-V(i+2,k)+8*V(i+1,k)-8*V(i-1,k)+V(i-2,k));
    end
end  
dx = dV;

for i=1:length(x)
    dx_true(i,:) = coupled_vdp_lorenz(0,x(i,:),par_vec);
end

figure('color','w','Position',[639,494,1069,484])
dim = 1;
Tspan = 10000;
time = linspace(dt,Tspan*dt,Tspan);
L = 1:Tspan;
subplot(2,2,1)
n_samples = 100;
plot(time(1:n_samples),x(1:n_samples,dim),'--','LineWidth',1.5)
hold on
plot(time(1:1:n_samples),x2(1:1:n_samples,dim),'--','LineWidth',0.5)
plot(time(1:n_samples),clean_x(1:n_samples,dim),'-.','LineWidth',1.5)
axis on
grid on
axis tight
xlabel('Time t'), ylabel('x')
set(gca,'FontSize',18,'FontWeight','bold')
set(gcf,'PaperPositionMode','auto')
title('x_1 coordinate','FontSize',20)

dim = 3;
subplot(2,2,2)
n_samples = 100;
plot(time(1:n_samples),x(1:n_samples,dim),'--','LineWidth',1.5)
hold on
plot(time(1:1:n_samples),x2(1:1:n_samples,dim),'--','LineWidth',0.5)
plot(time(1:n_samples),clean_x(1:n_samples,dim),'-.','LineWidth',1.5)
axis on
grid on
axis tight
xlabel('Time t'), ylabel('x')
set(gca,'FontSize',18,'FontWeight','bold')
set(gcf,'PaperPositionMode','auto')
title('x_3 coordinate','FontSize',20)
legend('True x','Noisy x','Clean x','FontSize',14,'Location','south')

dim = 1;
subplot(2,2,3)
plot(time,dx_true(L,dim),'LineWidth',1.5)
hold on
plot(time(3:Tspan-3),dx(3:Tspan-3,dim),'LineWidth',1.5)
axis on
grid on
axis tight
xlabel('Time t'), ylabel('dx/dt')
set(gca,'FontSize',18,'FontWeight','bold')
set(gcf,'PaperPositionMode','auto')
title('x_1 coordinate','FontSize',20)

dim = 3;
subplot(2,2,4)
plot(time,dx_true(L,dim),'LineWidth',1.5)
hold on
plot(time(3:Tspan-3),dx(3:Tspan-3,dim),'LineWidth',1.5)
axis on
grid on
axis tight
xlabel('Time t'), ylabel('dx/dt')
set(gca,'FontSize',18,'FontWeight','bold')
set(gcf,'PaperPositionMode','auto')
legend('True dx/dt','dx/dt estimates','FontSize',14,'Location','south')
title('x_3 coordinate','FontSize',20)
%% Plot best strategy
figure,
% Plot the attractor of fast VDP
L = 1:length(x);
plot(x(L,1),x(L,2),'Color',[.1 .1 .1],'LineWidth',0.5)
hold on
scatter(x(1,1),x(1,2),'filled','b')
axis on
grid on
axis tight
xlabel('x_1'), ylabel('x_2')
set(gca,'FontSize',10)
set(gcf,'Position',[100 100 600 400])
set(gcf,'PaperPositionMode','auto')
% plot(X(2:end,1),X(2:end,2),'o','MarkerSize',5,'Color',[0.75 0 0],...
%     'LineWidth',1.5,'MarkerFaceColor',[0.75 0 0])
figure,
% Plot the attractor of slow Lorenz
L = 1:length(x);
plot3(x(L,3),x(L,4),x(L,5),'Color',[.1 .1 .1],'LineWidth',0.5)
hold on
scatter3(x(1,3),x(1,4),x(1,5),'filled','b')
axis on
grid on
view(-5,12)
axis tight
xlabel('x'), ylabel('y'), zlabel('z')
set(gca,'FontSize',10)
set(gcf,'Position',[100 100 600 400])
set(gcf,'PaperPositionMode','auto')
plot3(X(2:end,3),X(2:end,4),X(2:end,5),'o','MarkerSize',3,'Color',[0.75 0 0],...
    'LineWidth',1.5)

%% Plot RL_strat
idx = 1;
X = zeros(length(RL_strat),n); % Initialize Ds
Ts_low = 0.001;
Ts_high = 0.64;
for i = 1:length(RL_strat)
    tmp = RL_strat(i);
    Ts = 0.5*Ts*(tmp == -1 && Ts > Ts_low) + Ts*(tmp == 0 || (tmp == -1 && Ts <= Ts_low) ...
        || (tmp == 1 && Ts >= Ts_high)) + 2*Ts*(tmp == 1 && Ts < Ts_high);
    idx = idx + Ts/dt;
    X(i,:) = x(idx,:);
end

figure,
% Plot the attractor of fast VDP
L = 1:length(x);
plot(x(L,1),x(L,2),'Color',[.1 .1 .1],'LineWidth',0.5)
hold on
scatter(x(1,1),x(1,2),'filled','b')
axis on
grid on
axis tight
xlabel('x_1'), ylabel('x_2')
set(gca,'FontSize',10)
set(gcf,'Position',[100 100 600 400])
set(gcf,'PaperPositionMode','auto')
plot(X(2:end,1),X(2:end,2),'o','MarkerSize',3,'Color',[0.75 0 0],...
    'LineWidth',1.5)
figure,
% Plot the attractor of slow Lorenz
L = 1:length(x);
plot3(x(L,3),x(L,4),x(L,5),'Color',[.1 .1 .1],'LineWidth',0.5)
hold on
scatter3(x(1,3),x(1,4),x(1,5),'filled','b')
axis on
grid on
view(-5,12)
axis tight
xlabel('x'), ylabel('y'), zlabel('z')
set(gca,'FontSize',10)
set(gcf,'Position',[100 100 600 400])
set(gcf,'PaperPositionMode','auto')
plot3(X(2:end,3),X(2:end,4),X(2:end,5),'o','MarkerSize',3,'Color',[0.75 0 0],...
    'LineWidth',1.5)
