close all, clear all, clc
%% Plot the best and RL sampling policies
% Initialize parameters
n = 4;
mu1 = 5;
mu2 = 4;
c1 = 0.005;
c2 = 1;
F = 5;
tau_fast = 0.2;
tau_slow = F*tau_fast;
x0 = [2,0,0,2];
beta = [mu1,mu2,tau_fast,tau_slow,c1,c2];
dt = 0.01; % delta t
%% Plot the data
Ts = 0.04;
Tmax = 10;
RL_strat =  ones(1,30);
best_strat = [3 3 3 3 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 3 3 1 3 2 1 3 2];
% Generate the data
% Integrate
dt = 0.0025*0.25;
tspan = [dt:dt:Tmax];
N = length(tspan);
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[t,x]=ode45(@(t,x) coupled_vdp(t,x,beta),tspan,x0,options);
% generate the trajectory data
tspan = [dt:dt:50];
N = length(tspan);
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[t,x]=ode45(@(t,x) coupled_vdp(t,x,beta),tspan,x0,options);
% Plot best_strat
idx = 1;
Ts_low = 0.001;
Ts_high = 0.64;
X = zeros(length(best_strat),n); % Initialize Ds
for i = 1:length(best_strat)
    tmp = best_strat(i);
    Ts = 0.5*Ts*(tmp == 1 && Ts > Ts_low) + Ts*(tmp == 2 || (tmp == 1 && Ts <= Ts_low) ...
        || (tmp == 3 && Ts >= Ts_high)) + 2*Ts*(tmp == 3 && Ts < Ts_high);
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
% Plot the attractor of slow VDP
L = 1:length(x);
plot(x(L,3),x(L,4),'Color',[.1 .1 .1],'LineWidth',0.5)
hold on
scatter(x(1,3),x(1,4),'filled','b')
axis on
grid on
axis tight
xlabel('x_3'), ylabel('x_4')
set(gca,'FontSize',10)
set(gcf,'Position',[100 100 600 400])
set(gcf,'PaperPositionMode','auto')
plot(X(2:end,3),X(2:end,4),'o','MarkerSize',3,'Color',[0.75 0 0],...
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
% Plot the attractor of slow VDP
L = 1:length(x);
plot(x(L,3),x(L,4),'Color',[.1 .1 .1],'LineWidth',0.5)
hold on
scatter(x(1,3),x(1,4),'filled','b')
axis on
grid on
axis tight
xlabel('x_3'), ylabel('x_4')
set(gca,'FontSize',10)
set(gcf,'Position',[100 100 600 400])
set(gcf,'PaperPositionMode','auto')
plot(X(2:end,3),X(2:end,4),'o','MarkerSize',3,'Color',[0.75 0 0],...
    'LineWidth',1.5)
