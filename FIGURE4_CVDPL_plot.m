clear all, close all, clc
%% PART 2: Coupled VDP - Lorenz system
clear all, clc
% Case 1: noise-free case (eta = 0)
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
x0 = [2 0 -8 8 27]';
Tmax = 200;
% Generate the data
% Integrate
dt = 0.004;
tspan = [dt:dt:Tmax];
par_vec = [mu,sigma,rho,beta,tau_fast,tau_slow,c1,c2];
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[t,x]=ode45(@(t,x) coupled_vdp_lorenz(t,x,par_vec),tspan,x0,options);

for i=1:length(x)
    dx_true(i,:) = coupled_vdp_lorenz(0,x(i,:),par_vec);
end
% Compute derivative using fourth order central difference
V = x;
dV = zeros(length(V)-5,n);
for i = 3:length(V)-3
    for k = 1:n
        dV(i-2,k) = (1/(12*dt))*(-V(i+2,k)+8*V(i+1,k)-8*V(i-1,k)+V(i-2,k));
    end
end  
dx(3:length(V)-3,:) = dV;
% Find the true Xi for Sindy algorithm
% Pool Data  (i.e., build library of nonlinear time series)
polyorder = 3;
Theta = poolData(x,n,polyorder);
m = size(Theta,2);
% Compute Sparse regression: sequential least squares
lambda = 0.1;      % lambda is our sparsification knob.
Xi_true = sparsifyDynamics(Theta,dx_true,lambda,n);

% Case 2: 0.1% NSR
NSR = 0.001; % Noise-to-signal ratio
% find the eta vale
eta = [];
for i = 1:n
    eta = sqrt(NSR*median(x(:,i).^2));
    x_01p(:,i) = x(:,i) + eta*randn(size(x,1),1);
end

% Case 3: 1% NSR
NSR = 0.01; % Noise-to-signal ratio
% find the eta vale
eta = [];
for i = 1:n
    eta = sqrt(NSR*median(x(:,i).^2));
    x_1p(:,i) = x(:,i) + eta*randn(size(x,1),1);
end
%% Filter the data and estimate the derivatives
% Fourier analysis                 
T = dt;             % Sampling period       
L = size(x,1);             % Length of signal
t = (0:L-1)*T;        % Time vector
fl = 0.5:0.1:20;
Fs = 1/dt;
for dim = 1:n
    % find optimal cut-off frequency
    RMSE = [];
    for k = 1:length(fl)
        [b,a] = butter(5,fl(k)/(Fs/2),'low');
        clean_x_tmp = filtfilt(b,a,x_01p(:,dim));
        RMSE(k) = sqrt(mean((clean_x_tmp-x(:,dim)).^2));
    end

    [~,idx] = min(RMSE);
    [b,a] = butter(5,fl(idx)/(Fs/2),'low');
    cx_01p(:,dim) = filtfilt(b,a,x_01p(:,dim));
    
    RMSE = [];
    for k = 1:length(fl)
        [b,a] = butter(5,fl(k)/(Fs/2),'low');
        clean_x_tmp = filtfilt(b,a,x_1p(:,dim));
        RMSE(k) = sqrt(mean((clean_x_tmp-x(:,dim)).^2));
    end

    [~,idx] = min(RMSE);
    [b,a] = butter(5,fl(idx)/(Fs/2),'low');
    cx_1p(:,dim) = filtfilt(b,a,x_1p(:,dim));
end

% Compute derivative using fourth order central difference
V = cx_01p;
dV = zeros(length(V)-5,n);
for i = 3:length(V)-3
    for k = 1:n
        dV(i-2,k) = (1/(12*dt))*(-V(i+2,k)+8*V(i+1,k)-8*V(i-1,k)+V(i-2,k));
    end
end  
dx_01p(3:length(V)-3,:) = dV;

V = cx_1p;
dV = zeros(length(V)-5,n);
for i = 3:length(V)-3
    for k = 1:n
        dV(i-2,k) = (1/(12*dt))*(-V(i+2,k)+8*V(i+1,k)-8*V(i-1,k)+V(i-2,k));
    end
end  
dx_1p(3:length(V)-3,:) = dV;
%% Plot the generated noisy data, filtered data, and, sampled points
Best1 = [3	3	3	1	3	3	3	2	2	1	1	1	3	3	3	1	2	2	1	1	3	2	2	2	2	1	1	3	2	2	3	1	3	2	2	2	2	2	3	3	2	1	2	2	2	2	3	2	2];

Best2 = [2	3	1	3	3	3	3	3	2	2	1	2	3	2	2	2	1	2	3	1	1	1	1	1	3	3	3	2	3	2	2	2	2	3	2	1	3	2	1	3	2	2	2	2	1	1	3	2	3	2	2	2	2	1	2	1	2	2	3	3	2	1	3	1	1	1	2	2	3	1	3	3	1	1	1	2	1];

Best3 = [3	3	3	3	2	3	2	2	2	2	2	2	2	2	2	2	2	1	2	2	1	2	3	3	1	2	2	2	1	2	2	2	2	1	2	2	2	3	3	1	3	3	2	2	2	1	3	2	2	1	3	1	1	2	2	2	2	2	3	3	1	1	1	2	1	2	3	3	3	3	1	2	1	2	3	2	2	3	2	1	1	3	3	1	1	3	2	3	2	2	2	2	2	1	1	2	2	3	1	3	1	3	1	1	2	1	1	1	3	2	2	2	2	2	2	3	1	2	1	1];
RL1 = [1	1	1	1	1	1	0	1	1	0	1	1	0	1	1	1	0	1	1	1	1	1	1	0	0	1	1	1	1	1	1	1	1	0	1	1	1	0	1	1	1	1	1	1	0	1	1	1	0	1	1	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0];
RL2 = [1	1	1	1	1	1	1	1	1	1	0	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0	1	0	0	0	0	1	1	0	1	1	-1	1	-1	1	0];
RL3 = [1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0	1	1	1	1	1	1	1	0	0	1	1	1	1	1	1	1	1	1	1	1	0	1	1	0	0	1	1	1	1	0	1	1	1	1	1	0	1	0	0	1	1	0	1	1	1	1	0	1	1	1	1	1	1	1	1	1	1	0	1	1	1];
idx = 1;
tnow = dt;
Ts_low = 0.001;
Ts_init = 0.016;
% The upper limit for Ts
Ts_high = 0.32;
Ts = Ts_init;
Xbest1 = []; Xbest2 = []; Xbest3 = [];
for i = 1:length(Best1)
    tmp = Best1(i);
    Ts = 0.5*Ts*(tmp == 1 && Ts > Ts_low) + Ts*(tmp == 2 || (tmp == 1 && Ts <= Ts_low) ...
        || (tmp == 3 && Ts >= Ts_high)) + 2*Ts*(tmp == 3 && Ts < Ts_high);
    idx = idx + Ts/dt;
    Xbest1 = [Xbest1; x(idx,:)];
end
idx = 1;
Ts = Ts_init;
for i = 1:length(Best2)
    tmp = Best2(i);
    Ts = 0.5*Ts*(tmp == 1 && Ts > Ts_low) + Ts*(tmp == 2 || (tmp == 1 && Ts <= Ts_low) ...
        || (tmp == 3 && Ts >= Ts_high)) + 2*Ts*(tmp == 3 && Ts < Ts_high);
    idx = idx + Ts/dt;
    Xbest2 = [Xbest2; cx_01p(idx,:)];
end
idx = 1;
Ts = Ts_init;
for i = 1:length(Best3)
    tmp = Best3(i);
    Ts = 0.5*Ts*(tmp == 1 && Ts > Ts_low) + Ts*(tmp == 2 || (tmp == 1 && Ts <= Ts_low) ...
        || (tmp == 3 && Ts >= Ts_high)) + 2*Ts*(tmp == 3 && Ts < Ts_high);
    if Ts < Ts_low
        Ts = Ts_low;
    end
    idx = idx + ceil(Ts/dt);
    Xbest3 = [Xbest3; cx_1p(idx,:)];
end
% RL
k = 1;
XRL1 = []; dXRL1 = []; 
XRL2 = []; dXRL2 = []; 
XRL3 = []; dXRL3 = []; 
for i = 1:k
    tnow = tnow + Ts_init;
    XRL1 = [XRL1; x(ceil(tnow/dt),:)];
    XRL2 = [XRL2; cx_01p(ceil(tnow/dt),:)];
    XRL3 = [XRL3; cx_1p(ceil(tnow/dt),:)];
    dXRL1 = [dXRL1; dx(ceil(tnow/dt),:)];
    dXRL2 = [dXRL2; dx_01p(ceil(tnow/dt),:)];
    dXRL3 = [dXRL3; dx_1p(ceil(tnow/dt),:)];
end
idx = (tnow/dt);
Ts = Ts_init;

for i = 1:length(RL1)
    tmp = RL1(i);
    Ts = 0.5*Ts*(tmp == -1 && Ts > Ts_low) + Ts*(tmp == 0 || (tmp == -1 && Ts <= Ts_low) ...
        || (tmp == 1 && Ts >= Ts_high)) + 2*Ts*(tmp == 1 && Ts < Ts_high);
    idx = idx + Ts/dt;
    XRL1 = [XRL1; x(idx,:)];
    dXRL1 = [dXRL1; dx(idx,:)];
end

% Evaluate the policy
% Xtmp = [x0'; XRL1];
% dXtmp = [coupled_vdp_lorenz(0,x0,par_vec)'; dXRL1];
% sample_size = length(Xtmp)
% train_time = dt*idx
% Theta = poolData(Xtmp,n,polyorder);
% Xi_hat = sparsifyDynamics(Theta,dXtmp,0.3,n);
% condnum = log(cond(Theta))
% Fnorm_error_last = norm(abs(Xi_true)-abs(Xi_hat),'fro')^2

idx = (tnow/dt);
Ts = Ts_init;
for i = 1:length(RL2)
    tmp = RL2(i);
    Ts = 0.5*Ts*(tmp == -1 && Ts > Ts_low) + Ts*(tmp == 0 || (tmp == -1 && Ts <= Ts_low) ...
        || (tmp == 1 && Ts >= Ts_high)) + 2*Ts*(tmp == 1 && Ts < Ts_high);
    idx = idx + Ts/dt;
    XRL2 = [XRL2; cx_01p(idx,:)];
    dXRL2 = [dXRL2; dx_01p(idx,:)];
end
% Evaluate the policy
% Xtmp = [x0'; XRL2];
% dXtmp = [coupled_vdp_lorenz(0,x0,par_vec)'; dXRL2];
% sample_size = length(Xtmp)
% train_time = dt*idx
% Theta = poolData(Xtmp,n,polyorder);
% Xi_hat = sparsifyDynamics(Theta,dXtmp ,0.5,n);
% condnum = log(cond(Theta))
% Fnorm_error_last = norm(abs(Xi_true)-abs(Xi_hat),'fro')^2

idx = (tnow/dt);
Ts = Ts_init;
for i = 1:length(RL3)
    tmp = RL3(i);
    Ts = 0.5*Ts*(tmp == -1 && Ts > Ts_low) + Ts*(tmp == 0 || (tmp == -1 && Ts <= Ts_low) ...
        || (tmp == 1 && Ts >= Ts_high)) + 2*Ts*(tmp == 1 && Ts < Ts_high);
    idx = idx + Ts/dt;
    XRL3 = [XRL3; cx_1p(idx,:)];
    dXRL3 = [dXRL3; dx_01p(idx,:)];
end
Xtmp = [x0'; XRL3];
dXtmp = [coupled_vdp_lorenz(0,x0,par_vec)'; dXRL3];
sample_size = length(Xtmp);
train_time = dt*idx;
Theta = poolData(Xtmp,n,polyorder);
Xi_hat = sparsifyDynamics(Theta,dXtmp ,0.5,n);
condnum = log(cond(Theta));
Fnorm_error_last = norm(abs(Xi_true)-abs(Xi_hat),'fro')^2;
figure('Position',[1,1,859,899])
subplot(3,2,1)
L = 1:5000;
n_samples = 30;
scatter(x(1,1),x(1,2),'filled')
hold on

plot(Xbest1(1:n_samples,1),Xbest1(1:n_samples,2),'o','MarkerSize',3,'Color',[0.4660 0.6740 0.1880],...
    'LineWidth',1.5)
plot(XRL1(1:n_samples,1),XRL1(1:n_samples,2),'s','MarkerSize',3,'Color',[0.75 0 0],...
    'LineWidth',1.5)
plot(x(L,1),x(L,2),'Color',[.1 .1 .1],'LineWidth',0.5)
axis on
grid on
axis tight
xlabel('x_1')
legend('Initial point','RBFS samples','RL-based samples','Location','northwest','Orientation','Horizontal')

set(gca,'FontWeight','bold','LineWidth',1.2,'FontSize',14);

set(gcf,'PaperPositionMode','auto')
title_labels = {'\color{blue}\fontsize{18}"Fast" VDP system', '\color{black}\fontsize{15}(a) '};
title(title_labels,'FontWeight','bold')
ylabels = {'\color{blue}\fontsize{18}NSR = 0%', '\color{black}\fontsize{15}x_2'};
ylabel(ylabels,'FontWeight','bold','Interpreter', 'tex')

% figure,
% L = 1:5000;
% n_samples = 50;
% plot(Xbest1(1,1),Xbest1(1,2),'o','MarkerSize',3,'Color',[0.4660 0.6740 0.1880],...
%     'LineWidth',1.5)
% plot(XRL1(1,1),XRL1(1,2),'s','MarkerSize',3,'Color',[0.75 0 0],...
%     'LineWidth',1.5)
% hold on
% scatter(x(1,1),x(1,2),'filled')
% plot(x(L,1),x(L,2),'Color',[.1 .1 .1],'LineWidth',0.5)
% plot(Xbest1(1:n_samples,1),Xbest1(1:n_samples,2),'o','MarkerSize',3,'Color',[0.4660 0.6740 0.1880],...
%     'LineWidth',1.5)
% plot(XRL1(1:n_samples,1),XRL1(1:n_samples,2),'s','MarkerSize',3,'Color',[0.75 0 0],...
%     'LineWidth',1.5)
% axis on
% grid on
% axis tight
% xlabel('x_1'), ylabel('x_2')
% set(gca,'FontSize',12,'FontWeight','bold')
% set(gcf,'PaperPositionMode','auto')
% title('"Fast" VDP system')
% legend('"Best" policy samples','DRL samples','FontSize',12,'Location','northwest')

subplot(3,2,3)
scatter(cx_01p(1,1),cx_01p(1,2),'filled')
hold on
plot(cx_01p(L,1),cx_01p(L,2),'Color',[.1 .1 .1],'LineWidth',0.5)
plot(Xbest2(1:n_samples,1),Xbest2(1:n_samples,2),'o','MarkerSize',3,'Color',[0.4660 0.6740 0.1880],...
    'LineWidth',1.5)
plot(XRL2(1:n_samples,1),XRL2(1:n_samples,2),'s','MarkerSize',3,'Color',[0.75 0 0],...
    'LineWidth',1.5)
axis on
grid on
axis tight
xlabel('x_1')
title('(c)')
set(gca,'FontWeight','bold','LineWidth',1.2,'FontSize',14);
set(gcf,'PaperPositionMode','auto')
ylabels = {'\color{blue}\fontsize{18}NSR = 0.1%', '\color{black}\fontsize{15}x_2'};
ylabel(ylabels,'FontWeight','bold','Interpreter', 'tex')

subplot(3,2,5)
scatter(cx_1p(1,1),cx_1p(1,2),'filled')
hold on
plot(cx_1p(L,1),cx_1p(L,2),'Color',[.1 .1 .1],'LineWidth',0.5)
plot(Xbest3(1:n_samples,1),Xbest3(1:n_samples,2),'o','MarkerSize',3,'Color',[0.4660 0.6740 0.1880],...
    'LineWidth',1.5)
plot(XRL3(1:n_samples,1),XRL3(1:n_samples,2),'s','MarkerSize',3,'Color',[0.75 0 0],...
    'LineWidth',1.5)
axis on
grid on
axis tight
xlabel('x_1'), title('(e)')
set(gca,'FontWeight','bold','LineWidth',1.2,'FontSize',14);
set(gcf,'PaperPositionMode','auto')
ylabels = {'\color{blue}\fontsize{18}NSR = 1%', '\color{black}\fontsize{15}x_2'};
ylabel(ylabels,'FontWeight','bold','Interpreter', 'tex')


subplot(3,2,2)
L = 1:5000;
scatter3(x(1,3),x(1,4),x(1,5),'filled')
hold on
plot3(x(L,3),x(L,4),x(L,5),'Color',[.1 .1 .1],'LineWidth',0.5)
plot3(Xbest1(1:n_samples,3),Xbest1(1:n_samples,4),Xbest1(1:n_samples,5),'o','MarkerSize',3,'Color',[0.4660 0.6740 0.1880],...
    'LineWidth',1.5)
plot3(XRL1(1:n_samples,3),XRL1(1:n_samples,4),XRL1(1:n_samples,5),'s','MarkerSize',3,'Color',[0.75 0 0],...
    'LineWidth',1.5)
axis on
grid on
view(-5,12)
axis tight
xlabel('x_3'), ylabel('x_4'), zlabel('x_5'), 
set(gca,'FontWeight','bold','LineWidth',1.2,'FontSize',14);
title_labels = {'\color{blue}\fontsize{18}"Slow" Lorenz system', '\color{black}\fontsize{15}(b) '};
title(title_labels,'FontWeight','bold')
set(gcf,'PaperPositionMode','auto')

L = 1:5000;
plot3(Xbest1(1,3),Xbest1(1,4),Xbest1(1,5),'o','MarkerSize',3,'Color',[0.4660 0.6740 0.1880],...
    'LineWidth',1.5)
hold on
plot3(XRL1(1,3),XRL1(1,4),XRL1(1,5),'s','MarkerSize',3,'Color',[0.75 0 0],...
    'LineWidth',1.5)
scatter3(x(1,3),x(1,4),x(1,5),'filled')
plot3(x(L,3),x(L,4),x(L,5),'Color',[.1 .1 .1],'LineWidth',0.5)
plot3(Xbest1(1:n_samples,3),Xbest1(1:n_samples,4),Xbest1(1:n_samples,5),'o','MarkerSize',3,'Color',[0.4660 0.6740 0.1880],...
    'LineWidth',1.5)
plot3(XRL1(1:n_samples,3),XRL1(1:n_samples,4),XRL1(1:n_samples,5),'s','MarkerSize',3,'Color',[0.75 0 0],...
    'LineWidth',1.5)
axis on
grid on
axis tight
view(-5,12)


subplot(3,2,4)
scatter3(cx_01p(1,3),cx_01p(1,4),cx_01p(1,5),'filled')
hold on
plot3(cx_01p(L,3),cx_01p(L,4),cx_01p(L,5),'Color',[.1 .1 .1],'LineWidth',0.5)
plot3(Xbest2(1:n_samples,3),Xbest2(1:n_samples,4),Xbest2(1:n_samples,5),'o','MarkerSize',3,'Color',[0.4660 0.6740 0.1880],...
    'LineWidth',1.5)
plot3(XRL2(1:n_samples,3),XRL2(1:n_samples,4),XRL2(1:n_samples,5),'s','MarkerSize',3,'Color',[0.75 0 0],...
    'LineWidth',1.5)
axis on
grid on
view(-5,12)
axis tight
xlabel('x_3'), ylabel('x_4'), zlabel('x_5'), title('(d)')
set(gca,'FontWeight','bold','LineWidth',1.2,'FontSize',14);
set(gcf,'PaperPositionMode','auto')

subplot(3,2,6)
scatter3(cx_1p(1,3),cx_1p(1,4),cx_1p(1,5),'filled')
hold on
plot3(cx_1p(L,3),cx_1p(L,4),cx_1p(L,5),'Color',[.1 .1 .1],'LineWidth',0.5)
plot3(Xbest3(1:n_samples,3),Xbest3(1:n_samples,4),Xbest3(1:n_samples,5),'o','MarkerSize',3,'Color',[0.4660 0.6740 0.1880],...
    'LineWidth',1.5)
plot3(XRL3(1:n_samples,3),XRL3(1:n_samples,4),XRL3(1:n_samples,5),'s','MarkerSize',3,'Color',[0.75 0 0],...
    'LineWidth',1.5)
axis on
grid on
view(-5,12)
axis tight
xlabel('x_3'), ylabel('x_4'), zlabel('x_5'), title('(f)')
set(gca,'FontWeight','bold','LineWidth',1.2,'FontSize',14);
set(gcf,'PaperPositionMode','auto')



