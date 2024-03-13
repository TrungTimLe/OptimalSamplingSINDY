clear all, close all, clc
n = 5;
rng(2)
% Initialize parameters
mu = 5;
c1 = 0.01;
c2 = 10;
F = 5;
sigma = 10; rho = 28; beta = 8/3;
tau_fast = 0.2;
tau_slow = F*tau_fast;
x0 = [2 0 -8 8 27]';

% Integrate
dt = 0.04;
tspan = [dt:dt:200];
par_vec = [mu,sigma,rho,beta,tau_fast,tau_slow,c1,c2];
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[t,x]=ode45(@(t,x) coupled_vdp_lorenz(t,x,par_vec),tspan,x0,options);
% Compute derivative using Total Variation Regularized Numerical Differentiation (TVDiff)
for i=1:length(x)
    dx_true(i,:) = coupled_vdp_lorenz(0,x(i,:),par_vec);
end
% Pool Data  (i.e., build library of nonlinear time series)
polyorder = 3;
Theta = poolData(x,n,polyorder);
m = size(Theta,2);

% Compute Sparse regression: sequential least squares
lambda = 0.1;      % lambda is our sparsification knob.
Xi_true = sparsifyDynamics(Theta,dx_true,lambda,n);
%% Generate data
dt = 0.004; % delta t
Tmax = 1000;
tspan = [dt:dt:Tmax];
[t,x]=ode45(@(t,x) coupled_vdp_lorenz(t,x,par_vec),tspan,x0,options);
% Generate noisy x
NSR = 0.001; % Noise-to-signal ratio
% find the eta vale
for i = 1:n
    eta = sqrt(NSR*median(x(:,i).^2));
    nx(:,i) = x(:,i) + eta*randn(size(x,1),1);
end
% Filter noise                
T = dt;             % Sampling period       
L = size(x,1);             % Length of signal
t = (0:L-1)*T;        % Time vector
fl = 1:0.1:20;
Fs = 1/dt;
for dim = 1:n
    % find optimal cut-off frequency
    RMSE = [];
    for k = 1:length(fl)
        [b,a] = butter(5,fl(k)/(Fs/2),'low');
        clean_x_tmp = filtfilt(b,a,nx(:,dim));
        RMSE(k) = sqrt(mean((clean_x_tmp-x(:,dim)).^2));
    end

    [~,idx] = min(RMSE);
    [b,a] = butter(5,fl(idx)/(Fs/2),'low');
    cx(:,dim) = filtfilt(b,a,nx(:,dim));
end
% cx = x; % Noise-free
% Compute derivative using fourth order central difference
V = cx;
dV = zeros(length(V)-5,n);
for i = 3:length(V)-3
    for k = 1:n
        dV(i-2,k) = (1/(12*dt))*(-V(i+2,k)+8*V(i+1,k)-8*V(i-1,k)+V(i-2,k));
    end
end  
dx(3:length(V)-3,:) = dV;
%% Greedy algorithm
range_val = 10*dt:5*dt:200*dt;
tol = 10;
lambda = 0.9;
n_iter = zeros(1,length(range_val));
cond_num = zeros(1,length(range_val));
log_trace = zeros(1,length(range_val));
time_t = zeros(1,length(range_val));
Fnorm_error_last = zeros(1,length(range_val));
for i = 1:length(range_val)
    X = x0';
    dX = coupled_vdp_lorenz(0,x0,par_vec)';
    tnow = 0;
    trace_max = []; 
    Fnorm_error = 100000;
    count = 0;
    while Fnorm_error >= tol && count <= 500
        count = count + 1;
        % find the data point to sample
        n_points = length(dt:dt:range_val(i));
        I_mat_trace = zeros(n_points,1); % information matrix trace
        idx = 0;
        for t = dt:dt:range_val(i)
            idx = idx + 1;
            % sample at t_now + t
            X_tmp = [X; cx(ceil((tnow+t)/dt+1),:)];
            Theta = poolData(X_tmp,n,polyorder);
            I_mat = inv(Theta'*Theta);% information matrix
            I_mat_trace(idx) = trace(I_mat);
        end
        % Greedy sampling 
        [~,m_idx] = max(I_mat_trace);
        tnow = tnow + m_idx*dt;
        X = [X; cx(ceil(tnow/dt+1),:)];
        dX = [dX; dx(ceil(tnow/dt+1),:)];
        Theta = poolData(X,n,polyorder);
        I_mat = inv(Theta*Theta');% information matrix
        if isnan(I_mat(1,1)) || isinf(I_mat(1,1))
            continue;
        end
        if isempty(find(isinf(I_mat(:)) == 1))
            trace_max(end+1) = trace(I_mat);
        else
            trace_max(end+1) = trace_max(end);
        end
        Xi_hat = sparsifyDynamics(Theta,dX,lambda,n);
        Fnorm_error = norm(abs(Xi_true)-abs(Xi_hat),'fro')^2;
    end
    Fnorm_error_last(i) = Fnorm_error;
    n_iter(i) = count;
    cond_num(i) = log(cond(Theta));
    log_trace(i) = log(real(trace(inv(Theta'*Theta))));
    time_t(i) = tnow;
end
%% Plot overall results
figure('Position',[680,50,976,946])
subplot(5,1,1)
semilogy(range_val,Fnorm_error_last)
ylabel('Fnorm error')
grid on
set(gca,'FontSize',14','FontWeight','bold')
subplot(5,1,2)
plot(range_val,n_iter)
grid on
ylabel('#iterations')
set(gca,'FontSize',14','FontWeight','bold')
subplot(5,1,3)
plot(range_val,cond_num)
grid on
ylabel('Log(\kappa)')
set(gca,'FontSize',14','FontWeight','bold')
subplot(5,1,4)
plot(range_val,log_trace)
grid on
ylabel('Log(tr(I))')
set(gca,'FontSize',14','FontWeight','bold')
subplot(5,1,5)
plot(range_val,time_t)
ylabel('Time t')
grid on
xlabel('search range')
set(gca,'FontSize',14','FontWeight','bold')
%% Plot the sampled data
% figure,
% % Plot the attractor
% L = 1:length(x);
% scatter3(x(1,1),x(1,2),x(1,3),'filled')
% hold on
% plot3(x(L,1),x(L,2),x(L,3),'Color',[.1 .1 .1],'LineWidth',0.5)
% axis on
% grid on
% view(-5,12)
% axis tight
% xlabel('x'), ylabel('y'), zlabel('z')
% set(gca,'FontSize',10)
% set(gcf,'Position',[100 100 600 400])
% set(gcf,'PaperPositionMode','auto')
% plot3(X(2:end,1),X(2:end,2),X(2:end,3),'o','MarkerSize',3,'Color',[0.75 0 0],...
%     'LineWidth',1.5)
% % legend('Intial condition')