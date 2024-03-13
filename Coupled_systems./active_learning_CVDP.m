clear all, close all, clc
%% Find the true Xi
%% Generate Data
polyorder = 3;
n = 4;
% Initialize parameters
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
% Change intial condition
x0_step = 300;
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
tspan = [dt:dt:x0_step*dt];
[t,x]=ode45(@(t,x) coupled_vdp(t,x,beta),tspan,x0,options);
x0 = x(end,:);
% Integrate
tspan = [dt:dt:20];
N = length(tspan);
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[t,x]=ode45(@(t,x) coupled_vdp(t,x,beta),tspan,x0,options);
eta = 0; % Noise level
% range = 0.1; % search range
tol = 0.001; % tolerance rate
opt = 1; % select optimality criterion
% eta = 0 -> range = 0.05, tol = [0.01 0.05], opt = 1,2,3
% eta = 0.1 -> range = 0.05, tol = [5 10], opt = 1,2,3
% eta = 0.5 -> range = 0.05, tol = [10 20], opt = 1,2,3
for i=1:length(x)
    noisy_x(i,:) = x(i,:) + eta*rand(1,n);
end
for i=1:length(x)
    dx(i,:) =  coupled_vdp(t,x(i,:),beta);
    noisy_dx(i,:) = coupled_vdp(0,x(i,:),beta)' + eta*rand(1,n);
end

% Pool Data  (i.e., build library of nonlinear time series)
Theta = poolData(x,n,polyorder);
m = size(Theta,2);

% Compute Sparse regression: sequential least squares
lambda = 0.1;      % lambda is our sparsification knob.
Xi_true = sparsifyDynamics(Theta,dx,lambda,n);
poolDataLIST({'x1','x2','x3','x4'},Xi_true,n,polyorder);
%% Active learning
range_val = 0.05:0.01:0.05;
n_iter = zeros(1,length(range_val));
Fnorm_error_last = zeros(1,length(range_val));
for i = 1:length(range_val)
    X = x0;
    dX = dx(1,:);
    tnow = 0;
    trace_max = []; 
    det_max = []; 
    eig_max = [];
    Fnorm_error = 100000;
    count = 0;
    while (Fnorm_error >= tol && count <= 100) || isnan(Fnorm_error)
        count = count + 1;
        % find the data point to sample
        n_points = length(dt:dt:range_val(i));
        I_mat_trace = zeros(n_points,1); % information matrix trace
        I_mat_det = zeros(n_points,1); % information matrix determinant
        I_mat_eig = zeros(n_points,1); % information matrix smallest eigenvalue
        idx = 0;
        for t = dt:dt:range_val(i)
            idx = idx + 1;
            % sample at t_now + t
            X_tmp = [X; noisy_x(int8((tnow+t)/dt+1),:)];
            Theta = poolData(X_tmp,n,polyorder);
            I_mat = inv(Theta'*Theta);% information matrix
            I_mat_trace(idx) = trace(I_mat);
            I_mat_det(idx) = det(I_mat);
            I_mat_eig(idx) = min(eig(I_mat));
        end
        % Greedy sampling and plot the distribution
        switch opt
        case 1 % A-optimality (trace)
%             figure,
            [~,m_idx] = max(I_mat_trace);
%             plot(I_mat_trace)
        case 2 % D-optimality 
            [~,m_idx] = max(I_mat_det);
%             figure,
    %         plot(I_mat_det)
        case 3 % E-optimality
            [~,m_idx] = max(I_mat_eig);
%             figure,
    %         plot(I_mat_eig)
        end
%         xlabel('Future time t')
%         close all
        tnow = tnow + m_idx*dt;
        X = [X; noisy_x(int8(tnow/dt+1),:)];
        dX = [dX; noisy_dx(int8(tnow/dt+1),:)];
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
end
%% Plot overall results
figure,
semilogy(trace_max)
hold on
semilogy(det_max)
semilogy(eig_max)
legend('A-optimality','D-optimality','E-optimality')
xlabel('Iteration'),ylabel('Value')
%% Plot the sampled data
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