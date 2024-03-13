clear all, close all, clc
%% Generate Data
Beta = [10; 28; 8/3]; % Lorenz's parameters (chaotic)
n = 3;
x0=[-8; 8; 27];  % Initial condition

% Integrate
dt = 0.04;
tspan=[dt:dt:20];
N = length(tspan);
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[t,x]=ode45(@(t,x) lorenz(t,x,Beta),tspan,x0,options);

%% Compute Derivative
for i=1:length(x)
    dx(i,:) = lorenz(0,x(i,:),Beta);
end

%% Pool Data  (i.e., build library of nonlinear time series)
polyorder = 3;
Theta = poolData(x,n,polyorder);
m = size(Theta,2);

%% Compute Sparse regression: sequential least squares
lambda = 0.1;      % lambda is our sparsification knob.
Xi_true = sparsifyDynamics(Theta,dx,lambda,n);
poolDataLIST({'x','y','z'},Xi_true,n,polyorder);
%% Simulate error
n_init = 4;
dx1 = dx(1:n_init,:);
idx = n_init;
n_data = length(x);
count = 0;
Fnorm_error = 100000;
while (idx < length(x) && Fnorm_error(end) > 0.01)
    idx = idx + 1;
    count = count + 1;
    dx1 = [dx1; dx(idx,:)];
    Xi_hat = sparsifyDynamics(Theta(1:idx,:),dx1,lambda,n);
    poolDataLIST({'x','y','z'},Xi_hat,n,polyorder);
%     de_dSdot = 2*(pinv(Theta(1:idx,:)))'*(pinv(Theta(1:idx,:))*dx1-Xi_true);
%     de_ds(count,1:3) = [-Beta(1)*de_dSdot(end,1) -de_dSdot(end,2) -Beta(3)*de_dSdot(end,3)]';
    Fnorm_error(count) = norm(Xi_true-Xi_hat,'fro')^2;
end


plot(Fnorm_error)
xlabel('Time t to which data are sampled up')
ylabel('Frobenius-norm error')
% figure,
% plot(de_ds(:,1))
% hold on
% plot(de_ds(:,2))
% plot(de_ds(:,3))
% legend('x','y','z')
%% Plot attractor
%%  Part 1: Attractor with inital data
figure
L = 1:length(x);
plot3(x(L,1),x(L,2),x(L,3),'Color',[.1 .1 .1],'LineWidth',1.5)
hold on
L = 1:(n_init+count);
plot3(x(L,1),x(L,2),x(L,3),'Color',[0.75 0 0],'LineWidth',1.5)
axis on
view(-5,12)
axis tight
xlabel('x'), ylabel('y'), zlabel('z')
set(gca,'FontSize',14)
set(gcf,'Position',[100 100 600 400])
set(gcf,'PaperPositionMode','auto')
legend('Full data','Initial data')

