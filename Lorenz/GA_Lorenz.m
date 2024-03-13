clear all, close all, clc
%% Find the true Xi
Beta = [10; 28; 8/3]; % Lorenz's parameters (chaotic)
n = 3; %number of state
x0 =[-8; 8; 27];  % Initial condition
% Change intial condition
x0_step = 300;
dt = 0.01; % delta t
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
% tspan = [dt:dt:x0_step*dt];
% [t,x]=ode45(@(t,x) lorenz(t,x,Beta),tspan,x0,options);
% x0 = x(end,:)';
tspan = [dt:dt:20];
N = length(tspan);
[t,x]=ode45(@(t,x) lorenz(t,x,Beta),tspan,x0,options);
for i=1:length(x)
    dx_true(i,:) = lorenz(0,x(i,:),Beta);
end
% Initialize parameters
NSR = 0.01; % Noise-to-signal ratio
tol = 1; % tolerance rate
% Pool Data  (i.e., build library of nonlinear time series)
polyorder = 3;
Theta = poolData(x,n,polyorder);
m = size(Theta,2);
% Compute Sparse regression: sequential least squares
lambda = 0.1;      % lambda is our sparsification knob.
Xi_true = sparsifyDynamics(Theta,dx_true,lambda,n);
%% Generate data for RL
rng(2)
n = 3; %number of state
x0 =[-8; 8; 27];  % Initial condition
dt = 0.004; % delta t
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
Tmax = 500;
tspan = [dt:dt:Tmax];
N = length(tspan);
[t,x] = ode45(@(t,x) lorenz(t,x,Beta),tspan,x0,options);
% Generate noisy x
% find the eta vale
for i = 1:n
    eta = sqrt(NSR*median(x(:,i).^2));
    nx(:,i) = x(:,i) + eta*randn(size(x,1),1);
end
% Filter noise                
T = dt;             % Sampling period       
L = size(x,1);             % Length of signal
t = (0:L-1)*T;        % Time vector
fl = 1:0.1:10;
Fs = 1/dt;
for dim = 1:3
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
range_val = 10*dt:5*dt:300*dt;
lambda = 0.1;
n_iter = zeros(1,length(range_val));
Fnorm_error_last = zeros(1,length(range_val));
for i = 1:length(range_val)
    X = x0';
    dX = lorenz(0,x0,Beta)';
    tnow = 0;
    trace_max = []; 
    det_max = []; 
    eig_max = [];
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
            det_max(end+1) = det(I_mat);
            eig_max(end+1) = min(eig(I_mat));
        else
            trace_max(end+1) = trace_max(end);
            det_max(end+1) = det_max(end);
            eig_max(end+1) = eig_max(end);
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